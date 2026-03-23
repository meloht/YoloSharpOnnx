using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx.Models
{
    public struct PreResult
    {
        public float[] OutData { get; set; }

        public int TopPad { get; set; }
        public int LeftPad { get; set; }
        public PreResult(float[] outData, int topPad, int leftPad)
        {
            OutData = outData;
            TopPad = topPad;
            LeftPad = leftPad;
        }
    }
}
