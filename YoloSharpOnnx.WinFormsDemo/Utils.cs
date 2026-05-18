using System;
using System.Collections.Generic;
using System.Management;
using System.Text;

namespace YoloSharpOnnx.WinFormsDemo
{
    internal class Utils
    {
        public static int GetMainGPU()
        {
            try
            {
                ManagementObjectSearcher searcher = new ManagementObjectSearcher("SELECT * FROM Win32_VideoController");
                int idx = 0;
                string[] set = ["NVIDIA", "GEFORCE", "AMD", "RADEON"];
                foreach (ManagementObject mo in searcher.Get())
                {
                    string name = mo["Name"]?.ToString() ?? "";
                    if (IsContain(name, set))
                    {
                        return idx;
                    }

                    string description = mo["Description"]?.ToString() ?? "";
                    if (IsContain(description, set))
                    {
                        return idx;
                    }
                    idx++;
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine(ex.Message);
            }
            return -1;
        }
        private static bool IsContain(string name, string[] set)
        {
            if (name != null)
            {
                foreach (var item in set)
                {
                    if (name.Contains(item))
                        return true;
                }
            }
            return false;
        }
    }
}
