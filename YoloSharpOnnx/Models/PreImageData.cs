using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx.Models
{
    public struct PreImageData
    {
        public Mat LetterboxImg { get; set; }
        public int TopPad { get; set; }
        public int LeftPad { get; set; }

        public PreImageData(Mat letterboxImg, int topPad, int leftPad)
        {
            LetterboxImg = letterboxImg;
            TopPad = topPad;
            LeftPad = leftPad;
        }
    }
}
