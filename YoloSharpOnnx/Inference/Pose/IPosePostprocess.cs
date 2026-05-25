using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;

namespace YoloSharpOnnx.Inference.Pose
{
    internal interface IPosePostprocess: IDisposable
    {
        List<PoseResult> PostProcessSync(OrtValue outputValue0, PreDetectResult preResult);
        List<PoseResult> PostProcessAsync(OrtValue outputValue0, PreDetectResult preResult);
    }
}
