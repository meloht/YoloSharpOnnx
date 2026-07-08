using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Channels;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Inference.DetectCore;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Detect
{
    internal class YoloDetectIoBinding : YoloDetectCoreIoBinding<DetectionResult, DetectionBatchResult>
    {
        public YoloDetectIoBinding(InferenceSession session, IDetCorePostprocess<DetectionResult> postprocess, IDetPreprocess preprocess, OnnxModel onnxModel, YoloConfig config)
            : base(session, postprocess, preprocess, onnxModel, config)
        {
        }

        protected override void DrawResults(Mat inputImage, List<DetectionResult> results)
        {
            YoloDrawResultUtils.DrawDetections(inputImage, results, _onnxModel.ColorPalette);
        }
    }
}
