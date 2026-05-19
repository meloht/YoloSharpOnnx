using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Classify.Models
{
    public class PreClsResultBatch : PreResultBatchBase, IDisposable
    {
        public PreClsResultBatch()
        {
        }
        public void Initialize(string imagePath, ImageBatchData data)
        {
            ImagePath = imagePath;
            Data = data;
        }

        public void Dispose()
        {
            ImagePath = null;
            Data = null;
        }
    }
}
