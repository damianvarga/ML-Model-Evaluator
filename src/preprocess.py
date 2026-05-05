from sklearn.model_selection import train_test_split


def split_data(df, target):
    # Normalize column names and target string to avoid whitespace/case issues
    df = df.copy()
    df.columns = df.columns.str.strip()
    target = str(target).strip()

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found. Available columns: {df.columns.tolist()}"
        )

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    return X_train, X_test, y_train, y_test
