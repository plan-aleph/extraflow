#  Copyright (c) 2023 Minato.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations
import numpy as np
import types

from abc import ABCMeta, abstractmethod
from typing import Any, TypeVar, Protocol, runtime_checkable, Generic, Type, Callable, Union, overload

from extraflow.core.type import InferenceInput, InferenceOutput
from extraflow.core.exceptions import InterfaceError

TInputType = TypeVar("TInputType", bound=InferenceInput)
TModelInputType = TypeVar("TModelInputType", bound=InferenceInput)
TModelOutputType = TypeVar("TModelOutputType", bound=InferenceOutput)
TOutputType = TypeVar("TOutputType", bound=InferenceOutput)


@runtime_checkable
class InferenceInterface(Protocol[TModelInputType, TModelOutputType]):
    def predict(
            self,
            model_input: TModelInputType
    ) -> TModelOutputType:
        ...


class BaseInferenceModel(
    Generic[
        TModelInputType,
        TModelOutputType
    ],
    metaclass=ABCMeta
):
    @abstractmethod
    def predict(
            self,
            model_input: TModelInputType
    ) -> TModelOutputType:
        ...


@runtime_checkable
class PreprocessInterface(
    Protocol[
        TInputType,
        TModelInputType
    ]
):
    def __call__(self, process_input: TInputType) -> TModelInputType:
        ...


class BasePreprocess(
    Generic[
        TInputType,
        TModelInputType
    ],
    metaclass=ABCMeta
):
    @abstractmethod
    def __call__(self, process_input: TInputType) -> TModelInputType:
        ...


@runtime_checkable
class PostprocessInterface(
    Protocol[
        TModelOutputType,
        TOutputType
    ]
):
    def __call__(self, process_input: TModelOutputType) -> TOutputType:
        ...


class BasePostprocess(
    Generic[
        TModelOutputType,
        TOutputType
    ],
    metaclass=ABCMeta
):
    @abstractmethod
    def __call__(self, process_input: TModelOutputType) -> TOutputType:
        ...


TInputOutput = TypeVar("TInputOutput", bound=Union[InferenceInput, InferenceOutput])


class EmptyProcess(BasePreprocess[TInputOutput, TInputOutput], BasePostprocess[TInputOutput, TInputOutput]):
    def __call__(self, process_input: TInputOutput) -> TInputOutput:
        return process_input


class BaseCombineProcess(metaclass=ABCMeta):
    def __init__(self, *args: Callable[[TInputOutput], TInputOutput]):
        self._process = args

    def __call__(self, process_input: TInputOutput) -> TInputOutput:
        for process in self._process:
            process_input = process(process_input)
        return process_input


class CombinePreprocess(BaseCombineProcess, BasePreprocess[TInputType, TModelInputType]):
    def __init__(self, *args: PreprocessInterface):
        for arg in args:
            if not isinstance(arg, PreprocessInterface):
                raise InterfaceError(PreprocessInterface, arg)
        super().__init__(*args)


class CombinePostprocess(BaseCombineProcess, BasePostprocess[TModelOutputType, TOutputType]):
    def __init__(self, *args: PostprocessInterface):
        for arg in args:
            if not isinstance(arg, PostprocessInterface):
                raise InterfaceError(PostprocessInterface, arg)
        super().__init__(*args)


class ExtraModel(
    Generic[
        TInputType,
        TModelInputType,
        TModelOutputType,
        TOutputType
    ],
    BaseInferenceModel[TModelInputType, TModelOutputType]
):
    @overload
    def __init__(self,
                 model: InferenceInterface[TModelInputType, TModelOutputType],
                 preprocess: list[PreprocessInterface] | PreprocessInterface | None = EmptyProcess(),
                 postprocess: list[PostprocessInterface] | PostprocessInterface | None = EmptyProcess(),
                 ):
        ...

    def __init__(
            self,
            model,
            preprocess=EmptyProcess(),
            postprocess=EmptyProcess(),
    ):
        assert isinstance(model, InferenceInterface), InterfaceError(InferenceInterface, model)
        if isinstance(preprocess, list):
            preprocess = CombinePreprocess(*preprocess)
        else:
            assert isinstance(preprocess, PreprocessInterface), InterfaceError(PreprocessInterface, preprocess)

        if isinstance(postprocess, list):
            postprocess = CombinePostprocess(*postprocess)
        else:
            assert isinstance(postprocess, PostprocessInterface), InterfaceError(PostprocessInterface, postprocess)

        self._model = model
        self._preprocess = preprocess
        self._postprocess = postprocess

    def predict(self, process_input: TInputType) -> TOutputType:
        return self._postprocess(
            self._model.predict(
                self._preprocess(
                    process_input
                )
            )
        )
