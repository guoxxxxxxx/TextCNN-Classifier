from classifier.engine.trainer import trainer

if __name__ == '__main__':
    # 如果想继续使用之前训练的模型训练的话，将model_path值更改为模型路径就行，比如说/runs/run_01/weights/best.pt
    # 如果为None则就是从0开始训练
    model_path = "runs/train/run_0/weights/best.pt"
    trainer(model_path=model_path)
