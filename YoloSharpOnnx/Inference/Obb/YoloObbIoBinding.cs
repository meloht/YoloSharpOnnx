using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Inference.DetectCore;
using YoloSharpOnnx.Inference.Pose;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Obb
{
    internal class YoloObbIoBinding: YoloDetectCoreIoBinding<ObbResult, ObbBatchResult>
    {
        public YoloObbIoBinding(InferenceSession session, SessionOptions options, IDetCorePostprocess<ObbResult> postprocess, IDetPreprocess preprocess, OnnxModel onnxModel, YoloConfig config)
            : base(session, options, postprocess, preprocess, onnxModel, config)
        {
        }

        protected override void DrawResults(Mat inputImage, List<ObbResult> results)
        {
            YoloUtils.DrawObbs(inputImage, results, _onnxModel.ColorPalette);
        }
    }
}
