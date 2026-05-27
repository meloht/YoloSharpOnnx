using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Text;
using System.Threading.Channels;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Classify.Models;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Inference.DetectCore;
using YoloSharpOnnx.Inference.Segment.Models;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Detect
{
    internal class YoloDetectOrtVal : YoloDetectCoreOrtVal<DetectionResult, DetectionBatchResult>
    {
        public YoloDetectOrtVal(InferenceSession session, SessionOptions options, IDetCorePostprocess<DetectionResult> postprocess, IDetPreprocess preprocess, OnnxModel onnxModel, YoloConfig config)
            : base(session, options, postprocess, preprocess, onnxModel, config)
        {
        }

        protected override void DrawResults(Mat inputImage, List<DetectionResult> results)
        {
            YoloDrawResultUtils.DrawDetections(inputImage, results, _onnxModel.ColorPalette);
        }

    }
}
