using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Inference;

namespace YoloSharpOnnx.Models
{
    public class ImageBatchData : IDisposable
    {

        public FixedBuffer FixedBuffer { get; set; }

        public OrtValue InputOrtValue { get; set; }

        public ImageBatchData(OnnxModel onnxModel)
        {

            FixedBuffer = new FixedBuffer(onnxModel.InputShapeSize);
            InputOrtValue = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance, TensorElementType.Float,
            onnxModel.InputShape, FixedBuffer.Address, onnxModel.InputSizeInBytes);
        }

        public void Dispose()
        {
            FixedBuffer?.Dispose();
            InputOrtValue?.Dispose();
        }
    }
}
