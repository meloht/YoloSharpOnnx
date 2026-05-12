using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Segment
{
    public class YoloSegOrtVal : YoloSegBase, IYoloSegment
    {
        public YoloSegOrtVal(InferenceSession session, SessionOptions options, ISegPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel, YoloConfig config) 
            : base(session, options, postprocess, preprocess, onnxModel, config)
        {
        }

        public List<SegResult> Run(Mat inputImage)
        {
            throw new NotImplementedException();
        }

        public YoloResult<SegResult> RunWithTime(Mat inputImage)
        {
            throw new NotImplementedException();
        }

        protected override void DisposedSub()
        {
            throw new NotImplementedException();
        }

        protected override List<SegResult> RunBatchInfer(PreDetectResultBatch preResult)
        {
            throw new NotImplementedException();
        }
    }
}
