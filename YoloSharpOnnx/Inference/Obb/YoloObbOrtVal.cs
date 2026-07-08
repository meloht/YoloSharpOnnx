using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Inference.DetectCore;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Obb
{
    internal class YoloObbOrtVal : YoloDetectCoreOrtVal<ObbResult, ObbBatchResult>
    {
        public YoloObbOrtVal(InferenceSession session, IDetCorePostprocess<ObbResult> postprocess, IDetPreprocess preprocess, OnnxModel onnxModel, YoloConfig config)
            : base(session, postprocess, preprocess, onnxModel, config)
        {
        }

        protected override void DrawResults(Mat inputImage, List<ObbResult> results)
        {
            YoloDrawResultUtils.DrawObbs(inputImage, results, _onnxModel.ColorPalette);
        }
    }
}
