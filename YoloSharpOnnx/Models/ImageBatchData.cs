using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Inference;

namespace YoloSharpOnnx.Models
{
    public class ImageBatchData : IDisposable
    {
        public Mat ResizedImg { get; set; }
        public FixedBuffer FixedBuffer { get; set; }

        public ImageBatchData(int len)
        {
            ResizedImg = new Mat();
            FixedBuffer = new FixedBuffer(len);
        }

        public void Dispose()
        {
            ResizedImg?.Dispose();
            FixedBuffer?.Dispose();
        }
    }
}
