using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference
{
    internal class PreResultBatchBase
    {
        public string ImagePath { get; set; }
        public ImageBatchData Data { get; set; }
    }
}
