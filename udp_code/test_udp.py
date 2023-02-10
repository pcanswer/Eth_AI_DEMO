import pytest
import udp_train
import pandas as pd
df = pd.read_csv('./udp_data/UDP_Process.csv')
df.iloc[0:2]
print(df)
# def test_resd_dbc():
#     read_dbc(file)

# def test_log_init():
#     log_init(file)
    
# def test_log_info():
#     log_info('111')


if __name__ == '__main__':
    pytest.main(["--cov", "--cov-report=html",'test_udp.py'])