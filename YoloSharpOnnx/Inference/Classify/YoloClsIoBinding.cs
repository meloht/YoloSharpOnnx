using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Classify
{
    public class YoloClsIoBinding : YoloClsBase, IYoloClassify
    {
        private bool disposedValue;
        private OrtIoBinding _binding;
        protected OrtValue _outputOrtValue;
        public YoloClsIoBinding(InferenceSession session, SessionOptions options, OnnxModel onnxModel, YoloConfig config, IClsPostprocess postprocess, IClsPreprocess preprocess) 
            : base(session, options, onnxModel, config, postprocess, preprocess)
        {
            _binding = _session.CreateIoBinding();

            _outputOrtValue = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance, TensorElementType.Float,
          _onnxModel.OutputShape, _outputFixedBuffer.Address, _onnxModel.OutputSizeInBytes);
            Warmup();
        }
        private void Warmup()
        {
            _binding.BindInput(_onnxModel.InputName, _inputOrtValue);
            _binding.BindOutput(_onnxModel.OutputName, _outputOrtValue);
            _binding.SynchronizeBoundInputs();

            _session.RunWithBinding(_runOptions, _binding);
            _binding.SynchronizeBoundOutputs();
        }
        public List<ClsResult> Run(Mat inputImage)
        {
            throw new NotImplementedException();
        }

        public YoloResult<ClsResult> RunWithTime(Mat inputImage)
        {
            throw new NotImplementedException();
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: dispose managed state (managed objects)
                }

                // TODO: free unmanaged resources (unmanaged objects) and override finalizer
                // TODO: set large fields to null
                DisposeCore();
                disposedValue = true;
            }
        }

        // // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
        // ~YoloClsIoBinding()
        // {
        //     // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
        //     Dispose(disposing: false);
        // }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
