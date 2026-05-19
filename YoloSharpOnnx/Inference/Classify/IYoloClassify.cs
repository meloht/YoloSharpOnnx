using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Classify.Models;
using YoloSharpOnnx.Inference.Detect.Models;

namespace YoloSharpOnnx.Inference.Classify
{
    public interface IYoloClassify: IDisposable
    {
        List<ClsResult> Run(Mat inputImage);

        YoloResult<ClsResult> RunWithTime(Mat inputImage);

        void DrawClassification(Mat img, List<ClsResult> results);

        ClsBatchResult[] BatchRunPostSync(List<string> listImg, IBatchProcessCallback<ClsBatchResult> processCallback, Action<ClsBatchResult> receiveAction);

        Task<ClsBatchResult[]> BatchRunAsyncPostSync(List<string> listImg, IBatchProcessCallback<ClsBatchResult> processCallback, Action<ClsBatchResult> receiveAction);

        IAsyncEnumerable<ClsBatchResult> BatchRunForeachSync(List<string> listImg);

        IYoloProcessAsync<PreClsResultBatch, ClsResult> GetYoloProcessAsync();

    }
}
