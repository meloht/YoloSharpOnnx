using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
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

namespace YoloSharpOnnx.Inference.Pose
{
    internal class YoloPoseIoBinding : YoloDetectCoreIoBinding<PoseResult, PoseBatchResult>
    {
        public YoloPoseIoBinding(InferenceSession session, SessionOptions options, IDetCorePostprocess<PoseResult> postprocess, IDetPreprocess preprocess, OnnxModel onnxModel, YoloConfig config) 
            : base(session, options, postprocess, preprocess, onnxModel, config)
        {
        }

        protected override void DrawResults(Mat inputImage, List<PoseResult> results)
        {
            YoloDrawResultUtils.DrawPoses(inputImage, results, _onnxModel.ColorPalette, _config);
        }
    }
}
