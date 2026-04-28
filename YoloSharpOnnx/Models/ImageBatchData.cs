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
        public Mat ResizedImg { get; set; }
        public FixedBuffer FixedBuffer { get; set; }

        public OrtValue InputOrtValue { get; set; }

        public ImageBatchData(OnnxModel onnxModel)
        {
            ResizedImg = new Mat();
            FixedBuffer = new FixedBuffer((int)onnxModel.InputShapeSize);
            InputOrtValue = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance, TensorElementType.Float,
            onnxModel.InputShape, FixedBuffer.Address, onnxModel.InputSizeInBytes);
        }

        public void Dispose()
        {
            ResizedImg?.Dispose();
            FixedBuffer?.Dispose();
            InputOrtValue?.Dispose();
        }
    }
}
