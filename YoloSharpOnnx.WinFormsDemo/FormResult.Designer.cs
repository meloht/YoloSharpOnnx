namespace YoloSharpOnnx.WinFormsDemo
{
    partial class FormResult
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            groupBox1 = new GroupBox();
            txtReuslt = new TextBox();
            groupBoxImage = new GroupBox();
            groupBox1.SuspendLayout();
            SuspendLayout();
            // 
            // groupBox1
            // 
            groupBox1.Controls.Add(txtReuslt);
            groupBox1.Dock = DockStyle.Top;
            groupBox1.Location = new Point(0, 0);
            groupBox1.Name = "groupBox1";
            groupBox1.Size = new Size(925, 77);
            groupBox1.TabIndex = 0;
            groupBox1.TabStop = false;
            groupBox1.Text = "Result";
            // 
            // txtReuslt
            // 
            txtReuslt.Dock = DockStyle.Fill;
            txtReuslt.Location = new Point(3, 19);
            txtReuslt.Multiline = true;
            txtReuslt.Name = "txtReuslt";
            txtReuslt.ReadOnly = true;
            txtReuslt.Size = new Size(919, 55);
            txtReuslt.TabIndex = 0;
            // 
            // groupBoxImage
            // 
            groupBoxImage.Dock = DockStyle.Fill;
            groupBoxImage.Location = new Point(0, 77);
            groupBoxImage.Name = "groupBoxImage";
            groupBoxImage.Size = new Size(925, 425);
            groupBoxImage.TabIndex = 1;
            groupBoxImage.TabStop = false;
            groupBoxImage.Text = "Result Image";
            // 
            // FormResult
            // 
            AutoScaleDimensions = new SizeF(7F, 17F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(925, 502);
            Controls.Add(groupBoxImage);
            Controls.Add(groupBox1);
            Name = "FormResult";
            Text = "FormResult";
            
            Load += FormResult_Load;
            groupBox1.ResumeLayout(false);
            groupBox1.PerformLayout();
            ResumeLayout(false);
        }

        #endregion

        private GroupBox groupBox1;
        private GroupBox groupBoxImage;
        private TextBox txtReuslt;
    }
}