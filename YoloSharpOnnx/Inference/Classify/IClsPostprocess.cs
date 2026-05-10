using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;

namespace YoloSharpOnnx.Inference.Classify
{
    public interface IClsPostprocess
    {
        List<ClsResult> PostProcess(OrtValue outputValue);
    }
}
