using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace YoloSharpOnnx.Inference
{
    public class OrtSafeMemoryHandle: SafeHandle
    {
        public OrtSafeMemoryHandle(IntPtr allocPtr) : base(allocPtr, true) { }

        public override bool IsInvalid => handle == IntPtr.Zero;

        public IntPtr Handle => handle;

        protected override bool ReleaseHandle()
        {
            Marshal.FreeHGlobal(handle);
            handle = IntPtr.Zero;
            return true;
        }
    }
}
