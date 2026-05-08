using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Inference.Detect;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Classify
{
    public class YoloClsBase : OnnxInferenceCore
    {
        protected readonly IClsPostprocess _postprocess;
        protected readonly IClsPreprocess _preprocess;
        public YoloClsBase(InferenceSession session, SessionOptions options, OnnxModel onnxModel, YoloConfig config, IClsPostprocess postprocess, IClsPreprocess preprocess) : base(session, options, onnxModel, config)
        {
            _postprocess = postprocess;
            _preprocess = preprocess;
        }
    }
}
