using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Pose
{
    public class PosePostprocessNMS : IPosePostprocess
    {
        public PosePostprocessNMS(OnnxModel onnx, YoloConfig yoloConfig)
        {
            
        }
        public List<PoseResult> PostProcessAsync(OrtValue outputValue0, PreDetectResult preResult)
        {
            throw new NotImplementedException();
        }

        public List<PoseResult> PostProcessSync(OrtValue outputValue0, PreDetectResult preResult)
        {
            throw new NotImplementedException();
        }
    }
}
