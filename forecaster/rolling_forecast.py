import os
import pandas as pd
import torch

def run_one_step_rolling_forecast(model, dataset, output_dir="previsoes", batch_size=1):
    """
    Rolling forecast SOMENTE PREVISÕES (sem valores true).
    'step' reflete posição real na série original.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            seq_x, _ = dataset[idx]                    # não precisamos mais do y aqui
            global_start = dataset.indices[idx]        # posição real na série
            
            batch_x = seq_x.unsqueeze(0).to(device)
            pred = model(batch_x)
            if isinstance(pred, tuple):
                pred = pred[0]
            pred_np = pred.squeeze(0).cpu().numpy()
            
            pred_df = pd.DataFrame(pred_np, columns=dataset.feature_columns)
            
            # Step reflete tempo real (posição global)
            real_start_step = global_start + dataset.lookback
            pred_df["step"] = range(real_start_step, real_start_step + len(pred_df))
            
            out_file = os.path.join(output_dir, f"janela_{idx:06d}.csv")
            pred_df.to_csv(out_file, index=False)
            if idx % 50 == 0 or idx == len(dataset)-1:
                print(f"Gerada previsão {idx+1}/{len(dataset)}")
    print(f"✅ Rolling forecast concluído! Arquivos salvos em: {output_dir}")