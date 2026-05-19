using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Detect.Models
{
    public class PreDetectResultBatch : PreResultBatchBase,IDisposable
    {
        public void Initialize(PreDetectResult preResult, string imagePath, ImageBatchData data)
        {
            PreResult = preResult;
            ImagePath = imagePath;
            Data = data;
        }
        public PreDetectResultBatch()
        {

        }
        public PreDetectResult PreResult { get; set; }


        public void Dispose()
        {
            ImagePath = null;
            Data = null;
        }
    }
}
