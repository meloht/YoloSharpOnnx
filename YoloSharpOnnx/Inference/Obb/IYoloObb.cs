using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;

namespace YoloSharpOnnx.Inference.Obb
{
    internal interface IYoloObb: IDisposable
    {
        List<ObbResult> Run(Mat inputImage);

        YoloResult<ObbResult> RunWithTime(Mat inputImage);

        void DrawObbs(Mat inputImage, List<ObbResult> list);

        ObbBatchResult[] BatchRun(List<string> listImg, IBatchProcessCallback<ObbBatchResult> processCallback, Action<ObbBatchResult> receiveAction);

        Task<ObbBatchResult[]> BatchRunAsync(List<string> listImg, IBatchProcessCallback<ObbBatchResult> processCallback, Action<ObbBatchResult> receiveAction);

        IAsyncEnumerable<ObbBatchResult> BatchRunForeachAsync(List<string> listImg);


        IYoloProcessAsync<PreDetectResultBatch, ObbResult> GetYoloProcessAsync();
    }
}
