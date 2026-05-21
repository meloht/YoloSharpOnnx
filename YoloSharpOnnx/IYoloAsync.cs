using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;

namespace YoloSharpOnnx
{
    public interface IYoloAsync : IDisposable
    {
        Task<List<DetectionResult>> RunDetectAsync(string inputImage);

        Task<List<DetectionResult>> RunDetectAsync(Mat img);

        Task<List<ClsResult>> RunClassifyAsync(string inputImage);

        Task<List<ClsResult>> RunClassifyAsync(Mat img);

        Task<List<SegResult>> RunSegmentAsync(string inputImage);

        Task<List<SegResult>> RunSegmentAsync(Mat img);

        Task<List<PoseResult>> RunPoseAsync(string inputImage);

        Task<List<PoseResult>> RunPoseAsync(Mat img);

    }
}
