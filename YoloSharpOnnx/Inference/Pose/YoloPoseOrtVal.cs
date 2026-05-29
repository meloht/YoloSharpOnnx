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
using YoloSharpOnnx.Inference.Pose;

using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Pose
{
    internal class YoloPoseOrtVal : YoloDetectCoreOrtVal<PoseResult, PoseBatchResult>
    {
        public YoloPoseOrtVal(InferenceSession session, SessionOptions options, IDetCorePostprocess<PoseResult> postprocess, IDetPreprocess preprocess, OnnxModel onnxModel, YoloConfig config)
            : base(session, options, postprocess, preprocess, onnxModel, config)
        {
        }

        protected override void DrawResults(Mat inputImage, List<PoseResult> results)
        {
            YoloDrawResultUtils.DrawPoses(inputImage, results, _onnxModel.ColorPalette, _config);
        }
    }
}
