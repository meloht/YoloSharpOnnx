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
        Task RunDetectAsync(Mat img, Guid guid, IBatchProcessCallback<DetectAsyncResult> callback, Action<DetectAsyncResult> receiveAction);
        Task RunDetectAsync(string inputImage, Guid guid, IBatchProcessCallback<DetectAsyncResult> callback, Action<DetectAsyncResult> receiveAction);

        Task<List<ClsResult>> RunClassifyAsync(string inputImage);

        Task<List<ClsResult>> RunClassifyAsync(Mat img);
        Task RunClassifyAsync(Mat img, Guid guid, IBatchProcessCallback<ClsAsyncResult> callback, Action<ClsAsyncResult> receiveAction);
        Task RunClassifyAsync(string inputImage, Guid guid, IBatchProcessCallback<ClsAsyncResult> callback, Action<ClsAsyncResult> receiveAction);

        Task<List<SegResult>> RunSegmentAsync(string inputImage);

        Task<List<SegResult>> RunSegmentAsync(Mat img);

        Task RunSegmentAsync(Mat img, Guid guid, IBatchProcessCallback<SegAsyncResult> callback, Action<SegAsyncResult> receiveAction);
        Task RunSegmentAsync(string inputImage, Guid guid, IBatchProcessCallback<SegAsyncResult> callback, Action<SegAsyncResult> receiveAction);

        Task<List<PoseResult>> RunPoseAsync(string inputImage);

        Task<List<PoseResult>> RunPoseAsync(Mat img);
        Task RunPoseAsync(Mat img, Guid guid, IBatchProcessCallback<PoseAsyncResult> callback, Action<PoseAsyncResult> receiveAction);
        Task RunPoseAsync(string inputImage, Guid guid, IBatchProcessCallback<PoseAsyncResult> callback, Action<PoseAsyncResult> receiveAction);

        Task<List<ObbResult>> RunObbDetectAsync(string inputImage);

        Task<List<ObbResult>> RunObbDetectAsync(Mat img);

        Task RunObbDetectAsync(Mat img, Guid guid, IBatchProcessCallback<ObbAsyncResult> callback, Action<ObbAsyncResult> receiveAction);
        Task RunObbDetectAsync(string inputImage, Guid guid, IBatchProcessCallback<ObbAsyncResult> callback, Action<ObbAsyncResult> receiveAction);

        Task CompleteAndCloseAsyncChannel();

    }
}
