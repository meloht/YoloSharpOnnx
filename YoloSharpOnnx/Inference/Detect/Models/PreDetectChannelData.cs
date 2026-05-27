using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;

namespace YoloSharpOnnx.Inference.Detect.Models
{

    internal class PreDetectChannelData<TAsyncResult, TBatchPreResult> : IGuidValue<TBatchPreResult>, IBatchPreChannelResult<TBatchPreResult, TAsyncResult> where TBatchPreResult : class
    {
        public Guid Guid { get; set; }
        public TBatchPreResult PreResult { get; set; }
        public IBatchProcessCallback<TAsyncResult> Callback { get; set; }
        public Action<TAsyncResult> ReceiveAction { get; set; }

        public void Initialize(Guid guid, TBatchPreResult preResult, IBatchProcessCallback<TAsyncResult> callback, Action<TAsyncResult> receiveAction)
        {
            Guid = guid;
            PreResult = preResult;
            Callback = callback;
            ReceiveAction = receiveAction;
        }
        public PreDetectChannelData()
        {
        }
        public void Dispose()
        {
            PreResult = null;
        }
    }
}
