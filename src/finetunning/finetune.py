from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    results = model.train(data="E:\\Skola\\FEI\\bakalarka\\bc-proj-opencv\\dataset\\data.yaml", epochs=100, imgsz=640, device=0, plots=True)
