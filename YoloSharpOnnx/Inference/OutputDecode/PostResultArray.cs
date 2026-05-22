using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloSharpOnnx.Inference.OutputDecode
{
    internal class PostResultArray : IDisposable
    {
        public readonly List<Rect> Boxes;
        public readonly List<float> Scores;
        public readonly List<int> ClassIds;
        public readonly List<int> Ids = null;

        public PostResultArray()
        {
            Boxes = new List<Rect>();
            Scores = new List<float>();
            ClassIds = new List<int>();
            Ids = null;
        }

        public PostResultArray(List<int> ids)
        {
            Ids = ids;
            Boxes = new List<Rect>();
            Scores = new List<float>();
            ClassIds = new List<int>();

        }

        public static PostResultArray CreateForDetect()
        {
            return new PostResultArray();
        }
        public static PostResultArray CreateForSegment()
        {
            return new PostResultArray(new List<int>());
        }

        public void Dispose()
        {
            Boxes.Clear();
            Scores.Clear();
            ClassIds.Clear();
            Ids?.Clear();
        }
    }
}
