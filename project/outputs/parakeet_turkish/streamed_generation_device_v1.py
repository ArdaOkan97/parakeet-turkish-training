"""Expose GPU execution for generation when frozen parameter storage is CPU.

Transformers infers model.device from the first parameter. Frozen offloaded
encoder storage comes first, so generic generation otherwise creates CPU
decoder IDs even though the decoder executes on GPU. Only this instance gets
an inherited class with a corrected device property; forward code is unchanged.
"""
import torch


class _GPUDecoderExecution:
    @property
    def device(self):
        return self.decoder.embedding.weight.device


def apply(model):
    if not hasattr(model,'_diagnostic_frozen_backing'):raise ValueError('Declared streamed model required')
    if model.decoder.embedding.weight.device.type!='cuda':raise ValueError('Actual GPU decoder required')
    original=type(model)
    model.__class__=type('Streamed'+original.__name__,(_GPUDecoderExecution,original),{'__module__':__name__})
    if model.device.type!='cuda':raise ValueError('Generation execution device did not bind to actual decoder')
    return {'kind':'streamed_model_generation_execution_device_v1','original_class':original.__name__,
            'execution_device':str(model.device),'forward_implementation_changed':False,
            'all_parameters_and_dtypes_preserved':True,'original_global_class_modified':False}
