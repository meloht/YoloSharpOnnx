using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;

namespace YoloSharpOnnx.Inference.Pose
{
    internal interface IYoloPose: IDisposable
    {
        List<PoseResult> Run(Mat inputImage);

        YoloResult<PoseResult> RunWithTime(Mat inputImage);

        void DrawPoses(Mat inputImage, List<PoseResult> list);

        PoseBatchResult[] BatchRun(List<string> listImg, IBatchProcessCallback<PoseBatchResult> processCallback, Action<PoseBatchResult> receiveAction);

        Task<PoseBatchResult[]> BatchRunAsync(List<string> listImg, IBatchProcessCallback<PoseBatchResult> processCallback, Action<PoseBatchResult> receiveAction);

        IAsyncEnumerable<PoseBatchResult> BatchRunForeachAsync(List<string> listImg);


        IYoloProcessAsync<PreDetectResultBatch, PoseResult> GetYoloProcessAsync();
    }
}
