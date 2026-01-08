# -*- coding: utf-8 -*-
"""
report_generator.py

Haftalık veya günlük performans raporları oluşturmak için kullanılabilecek
bir yardımcı araç. Bu script, ``trade_log.json`` dosyasını okuyarak
 temel metrikleri (kazanç/zarar, win‑rate, en büyük düşüş, işlem
sayısı) hesaplar ve bunları basit bir Markdown rapor olarak yazdırır.

Kullanım:
    (venv) python report_generator.py --output report.md

Opsiyonel parametreler ile belirli bir tarih aralığı, sembol filtresi
veya metrikler belirlenebilir. Varsayılan olarak tüm veriyi raporlar.
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
import argparse
import pandas as pd


def generate_report(trade_log_path: Path, output_path: Path, start_date: str | None = None, end_date: str | None = None) -> None:
    """Trade log dosyasını okuyup performans raporu üret.

    Args:
        trade_log_path: trade_log.json dosyasının yolu.
        output_path: Yazılacak rapor (Markdown) dosyası.
        start_date: ISO tarih (YYYY‑MM‑DD) başlangıç filtresi.
        end_date: ISO tarih (YYYY‑MM‑DD) bitiş filtresi.
    """
    if not trade_log_path.exists():
        print(f"[report_generator] trade_log.json bulunamadı: {trade_log_path}")
        return
    try:
        with trade_log_path.open('r', encoding='utf-8') as f:
            trades = json.load(f)
    except Exception as e:
        print(f"[report_generator] trade_log.json okunamadı: {e}")
        return
    if not isinstance(trades, list):
        print("[report_generator] trade_log.json beklenen formatta değil.")
        return
    # JSON listeden DataFrame
    df = pd.DataFrame(trades)
    if df.empty:
        print("[report_generator] Raporlanacak veri yok.")
        return
    # Tarih filtresi uygula
    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date)
            df = df[df['timestamp_open'] >= start_dt.isoformat()]
        except Exception:
            pass
    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date)
            df = df[df['timestamp_open'] <= end_dt.isoformat()]
        except Exception:
            pass
    # PnL hesapla (varsayım: pnl_abs alanı var ve USDT cinsinden)
    total_pnl = df['pnl_abs'].sum() if 'pnl_abs' in df else 0.0
    win_trades = df[df['pnl_abs'] > 0].shape[0]
    lose_trades = df[df['pnl_abs'] <= 0].shape[0]
    total_trades = df.shape[0]
    win_rate = (win_trades / total_trades) * 100 if total_trades > 0 else 0.0
    # En büyük geri çekilme (max drawdown) sütunu varsa
    max_dd = df['max_drawdown_pct'].max() if 'max_drawdown_pct' in df else None
    # Rapora yaz
    lines = []
    lines.append(f"# Performans Raporu\n")
    lines.append(f"Toplam İşlem Sayısı: {total_trades}")
    lines.append(f"Toplam PnL (USDT): {total_pnl:.2f}")
    lines.append(f"Kazanan İşlem Sayısı: {win_trades}")
    lines.append(f"Kaybeden İşlem Sayısı: {lose_trades}")
    lines.append(f"Win‑Rate: {win_rate:.2f}%")
    if max_dd is not None:
        lines.append(f"En Büyük Çekilme (Max DD %): {max_dd:.2f}")
    lines.append("\n## Sembol Bazlı Özet\n")
    if 'symbol' in df:
        grouped = df.groupby('symbol')['pnl_abs'].sum().reset_index()
        for _, row in grouped.iterrows():
            lines.append(f"* {row['symbol']}: {row['pnl_abs']:.2f} USDT toplam PnL")
    # Yaz
    try:
        output_path.write_text('\n'.join(lines), encoding='utf-8')
        print(f"[report_generator] Rapor oluşturuldu: {output_path}")
    except Exception as e:
        print(f"[report_generator] Rapor yazılamadı: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trade performans raporu oluştur.')
    parser.add_argument('--output', type=str, default='trade_report.md', help='Yazılacak rapor dosyası')
    parser.add_argument('--start', type=str, default=None, help='Başlangıç tarihi (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='Bitiş tarihi (YYYY-MM-DD)')
    args = parser.parse_args()
    trade_log_path = Path('trade_log.json')
    out_path = Path(args.output)
    generate_report(trade_log_path, out_path, args.start, args.end)