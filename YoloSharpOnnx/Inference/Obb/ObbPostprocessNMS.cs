using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Obb
{
    internal class ObbPostprocessNMS: IObbPostprocess
    {
        public ObbPostprocessNMS(OnnxModel onnx, YoloConfig yoloConfig)
        {
            
        }

        public void Dispose()
        {
            throw new NotImplementedException();
        }

        public List<ObbResult> PostProcessAsync(OrtValue output, PreDetectResult preResult)
        {
            throw new NotImplementedException();
        }

        public List<ObbResult> PostProcessSync(OrtValue output, PreDetectResult preResult)
        {
            throw new NotImplementedException();
        }
    }
}
