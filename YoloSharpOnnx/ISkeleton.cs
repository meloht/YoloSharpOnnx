using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx
{
    public interface ISkeleton
    {
        int ConnectionCount { get; }

        Scalar GetKeypointColor(int index);

        Scalar GetLineColor(int index);

        PosePoint GetKeypoint1(int index, PosePoint[] keyPoints);
        PosePoint GetKeypoint2(int index, PosePoint[] keyPoints);
    }
}
