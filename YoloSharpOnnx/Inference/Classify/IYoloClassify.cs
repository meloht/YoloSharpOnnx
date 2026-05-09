using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;

namespace YoloSharpOnnx.Inference.Classify
{
    public interface IYoloClassify: IDisposable
    {
        List<ClsResult> Run(Mat inputImage);

        YoloResult<ClsResult> RunWithTime(Mat inputImage);

        void DrawClassification(Mat img, List<ClsResult> results);

        ClsBatchResult[] BatchCls(List<string> listImg, IBatchProcessCallback<ClsBatchResult> processCallback, Action<ClsBatchResult> receiveAction);

        Task<ClsBatchResult[]> BatchClsAsync(List<string> listImg, IBatchProcessCallback<ClsBatchResult> processCallback, Action<ClsBatchResult> receiveAction);

        IAsyncEnumerable<ClsBatchResult> BatchClsForeachAsync(List<string> listImg);
    }
}
