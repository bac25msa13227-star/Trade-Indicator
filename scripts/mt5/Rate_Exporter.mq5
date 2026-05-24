//+------------------------------------------------------------------+
//| Rate_Exporter.mq5                                                |
//| Streams Strategy Tester OHLC bars to Common\Files as CSV.         |
//+------------------------------------------------------------------+
#property strict
#property version "1.01"

input string InpOutFile = "mt5_rates_export.csv";

int g_handle = INVALID_HANDLE;
datetime g_last_written = 0;

bool WriteBar(const int shift) {
    datetime bar_time = iTime(_Symbol, _Period, shift);
    if (bar_time <= 0 || bar_time == g_last_written) {
        return false;
    }
    double open = iOpen(_Symbol, _Period, shift);
    double high = iHigh(_Symbol, _Period, shift);
    double low = iLow(_Symbol, _Period, shift);
    double close = iClose(_Symbol, _Period, shift);
    long tick_volume = iVolume(_Symbol, _Period, shift);
    int spread = (int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
    if (open <= 0 || high <= 0 || low <= 0 || close <= 0) {
        return false;
    }
    FileWrite(
        g_handle,
        TimeToString(bar_time, TIME_DATE | TIME_MINUTES),
        DoubleToString(open, _Digits),
        DoubleToString(high, _Digits),
        DoubleToString(low, _Digits),
        DoubleToString(close, _Digits),
        tick_volume,
        spread,
        0
    );
    g_last_written = bar_time;
    return true;
}

int OnInit() {
    g_handle = FileOpen(InpOutFile, FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON, ',');
    if (g_handle == INVALID_HANDLE) {
        PrintFormat("FileOpen failed for %s, err=%d", InpOutFile, GetLastError());
        return INIT_FAILED;
    }
    FileWrite(g_handle, "time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume");
    PrintFormat("RATE_EXPORT_START file=%s", InpOutFile);
    return INIT_SUCCEEDED;
}

void OnTick() {
    if (g_handle == INVALID_HANDLE) {
        return;
    }
    WriteBar(1);
}

void OnDeinit(const int reason) {
    if (g_handle != INVALID_HANDLE) {
        WriteBar(0);
        FileClose(g_handle);
        g_handle = INVALID_HANDLE;
        PrintFormat("RATE_EXPORT_DONE file=%s last_time=%s reason=%d", InpOutFile, TimeToString(g_last_written, TIME_DATE | TIME_MINUTES), reason);
    }
}
