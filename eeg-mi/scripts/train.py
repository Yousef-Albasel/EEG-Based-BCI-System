def main(data_path, save_dir):
    import pickle
    from utils.train_utils import train
    with open(data_path, "rb") as f:
        X_train, y_train, X_val, y_val, _ = pickle.load(f)
    train(X_train, y_train, X_val, y_val, save_dir=save_dir)

if __name__ == "__main__":
    main()
