import os
import pandas as pd
import torch

def run_one_step_rolling_forecast(model, dataset, output_dir="previsoes", batch_size=1):
    """
    Rolling forecast 1-step-ahead no período de teste.
    Compatível com MultiVariateTimeSeriesDataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            seq_x, seq_y = dataset[idx]
            batch_x = seq_x.unsqueeze(0).to(device)          # [1, lookback, n_features]
            
            pred = model(batch_x)                            # (1, pred_len, n_features)
            
            # Tratamento robusto da saída do modelo
            if isinstance(pred, tuple):
                pred = pred[0]
            pred = pred.squeeze(0).cpu()                     # [pred_len, n_features]
            
            pred_np = pred.numpy()
            true_np = seq_y[-model.pred_len:, :].cpu().numpy() if hasattr(model, 'pred_len') else seq_y.numpy()
            
            pred_df = pd.DataFrame(pred_np, columns=dataset.feature_columns)
            true_df = pd.DataFrame(true_np, columns=dataset.feature_columns)
            
            pred_df["step"] = range(1, len(pred_df) + 1)
            true_df["step"] = range(1, len(true_df) + 1)
            
            out_df = pred_df.copy()
            for col in dataset.feature_columns:
                out_df[f"true_{col}"] = true_df[col]
            
            # Metadata (opcional)
            meta_window = dataset.get_metadata_window(idx)
            if meta_window is not None and len(meta_window) > 0:
                last_meta = meta_window.iloc[-1].to_dict()
                for mk, mv in last_meta.items():
                    out_df[f"meta_{mk}"] = mv
            
            out_file = os.path.join(output_dir, f"janela_{idx:06d}.csv")
            out_df.to_csv(out_file, index=False)
            
            if idx % 100 == 0:
                print(f"Processada janela {idx}/{len(dataset)}")
    
    print(f"✅ Rolling forecast concluído! Arquivos salvos em: {output_dir}")