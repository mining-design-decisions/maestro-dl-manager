"""
This module provides special functionality for
handling configuration options in a centralized
manner.

It provides an easy to configure way to set up
an elaborate, automatically handled command
line interface
"""

from __future__ import annotations

import itertools

##############################################################################
##############################################################################
# Imports
##############################################################################

import fastapi
import requests
import uvicorn

from .config.arguments import Argument
from .config.constraints import Constraint
from .config.core import ConfigFactory, Config
from .config.parsing import ArgumentListParser, ArgumentConsumer, ArgumentParsingError
from . import logger
from . import checkpointing

log = logger.get_logger("App Builder")


##############################################################################
##############################################################################
# Web API builder
##############################################################################


class WebApp:
    def __init__(self, spec):
        self._app = fastapi.FastAPI(root_path="/dl-manager")
        self._router = fastapi.APIRouter()
        self._callbacks = {}
        self._setup_callbacks = []
        self._endpoints = {}
        self._config_factory = ConfigFactory()
        self._spec = spec
        self._register_system_properties()
        self._build_endpoints(self._spec['commands'].values())
        self._add_static_endpoints()

    def _register_system_properties(self):
        self._config_factory.register_namespace("system.storage")
        self._config_factory.register_namespace("system.security")
        self._config_factory.register_namespace("system.os")
        self._config_factory.register_namespace("system.resources")
        self._config_factory.register_namespace("system.management")

        # Variable for tracking the amount of available threads
        self._config_factory.register("system.resources.threads")

        # Database credentials
        # self._config_factory.register('system.security.db-username')
        # self._config_factory.register('system.security.db-password')
        self._config_factory.register("system.security.db-token")

        # Control of self-signed certificates
        self._config_factory.register("system.security.allow-self-signed-certificates")

        # Database connection
        self._config_factory.register("system.storage.database-url")
        self._config_factory.register("system.storage.database-api")

        # Model Saving Support
        # - system.storage.generators
        #       A list of filenames pointing to the files containing
        #       the configurations for the trained feature generators.
        # - system.storage.auxiliary
        #       A list of filenames containing auxiliary files which
        #       must be included in the folder for saving a
        #       pretrained model.
        # - system.storage.auxiliary_map
        #       A mapping which is necessary to resolve filenames
        #       when loading a pretrained model.
        #       It maps "local" filenames to the actual location
        #       in the folder containing the pretrained model.
        # - system.storage.file_prefix
        #       A prefix which _should_ be used for all files
        #       generated by the pipeline, which is often
        #       forgotten about in practice.
        self._config_factory.register("system.storage.generators")
        self._config_factory.register("system.storage.auxiliary")
        self._config_factory.register("system.storage.auxiliary_map")
        self._config_factory.register("system.storage.file_prefix")

        # Current system state
        self._config_factory.register("system.management.active-command")
        self._config_factory.register("system.management.app")

        # Target home and data directories.
        self._config_factory.register("system.os.peregrine")
        self._config_factory.register("system.os.home-directory")
        self._config_factory.register("system.os.data-directory")
        self._config_factory.register("system.os.scratch-directory")

    def _build_endpoints(self, commands):
        for command in commands:
            self._build_endpoint(command)

    def _build_endpoint(self, spec):
        endpoint = _Endpoint(spec, self._config_factory, self.dispatch)
        if not endpoint.private:
            self._router.post("/" + endpoint.name, description=endpoint.description)(
                endpoint.invoke
            )
        self._endpoints[endpoint.name] = endpoint

    def _add_static_endpoints(self):
        @self._router.get("/endpoints")
        async def get_endpoints():
            result = {
                'name': self._spec['name'],
                'help': self._spec['help'],
                'commands': {
                    key: {
                        'name': value['name'],
                        'help': value['help'],
                        'args': {
                            k: v.get_json_spec() for k, v in value['args'].items()
                        },
                        # 'constraints': [
                        #     x.get_json_spec() for x in value['constraints']
                        # ]
                        'constraints': list(
                            itertools.chain(
                                *(x.get_json_spec() for x in value['constraints'])
                            )
                        )
                    }
                    for key, value in self._spec['commands'].items()
                }
            }
            for key, value in self._spec['commands'].items():
                print(value['constraints'])
            return result

    def register_callback(self, event, func):
        self._callbacks[event] = func

    def register_setup_callback(self, func):
        self._setup_callbacks.append(func)

    def deploy(self, port):
        self._app.include_router(self._router)
        uvicorn.run(
            self._app,
            host="0.0.0.0",
            port=port,
        )

    def execute_script(self, filename, *, invalidate_checkpoints):
        manager = checkpointing.CheckpointManager(filename)
        if invalidate_checkpoints:
            manager.invalidate()
        token = manager.get_auth()["token"]
        for command in manager.commands():
            # Refresh the token
            response = requests.post(
                url=manager.get_auth()["token-endpoint"],
                headers={"Authorization": "Bearer " + token},
            )
            response.raise_for_status()
            token = response.json()["access_token"]
            # Call the endpoint internally
            endpoint = command["cmd"]
            payload = {"auth": {"token": token}, "config": command["args"]}
            self._endpoints[endpoint].invoke_with_json(payload)

    def invoke_endpoint(self, name: str, conf: Config, payload):
        return self._endpoints[name].run(conf, payload)

    def dispatch(self, name, conf: Config):
        conf.set("system.management.active-command", name)
        conf.set("system.management.app", self)
        for callback in self._setup_callbacks:
            callback(conf)
        return self._callbacks[name](conf)

    def new_config(self, *namespaces) -> Config:
        return self._config_factory.build_config(*namespaces)


class _Endpoint:

    def __init__(self, spec, config_factory: ConfigFactory, callback):
        self._spec = spec
        self.name = self._spec['name']
        self.private = self._spec['private']
        self.description = self._spec['help']
        self._config_factory = config_factory
        log.info(f'Registering argument namespace: {self.name}')
        config_factory.register_namespace(self.name)
        for name in self._spec['args']:
            config_factory.register(f'{self.name}.{name}')
            log.info(f'Registered argument: {self.name}.{name}')
        self._dispatcher = callback
        self._parser = ArgumentListParser.from_spec_dict(
            self.name,
            {self.name: (self._spec['args'], self._spec['constraints'])},
            multi_valued=False,
            tunable_arguments=False,
            logger=log
        )

    async def invoke(self, req: fastapi.Request):
        if self.private:
            raise fastapi.HTTPException(
                    detail=f"Endpoint {self.name} is private/internal", status_code=406
                )
        payload = await req.json()
        return self.invoke_with_json(payload)

    def invoke_with_json(self, payload):
        conf = self._config_factory.build_config(self.name, "system")
        if "auth" in payload:
            auth = payload["auth"]
            # conf.set('system.security.db-username', auth['username'])
            # conf.set('system.security.db-password', auth['password'])
            conf.set("system.security.db-token", auth["token"])
        return self.run(conf, payload["config"])

    def run(self, conf: Config, payload):
        try:
            args = self._parser.validate({f'{self.name}.0': payload})[f'{self.name}'][0]
        except ArgumentParsingError as e:
            raise fastapi.HTTPException(
                detail=e.message,
                status_code=400 if e.is_user_error else 500
            )
        for name, value in args.items():
            conf.set(f"{self.name}.{name}", value)
        return self._dispatcher(self.name, conf)
