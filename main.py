#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 23:37:03 2022

@author: kotaseshimo
"""


import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import generate_sample
#from mpl_toolkits.mplot3d import Axes3D
from numpy import matlib

#number_of_generating_samples = 10000  # 生成するサンプル数
#desired_sum_of_components = 1 # 合計を指定する特徴量がある場合の、合計の値。例えば、この値を 100 にすれば、合計を 100 にできます

def generating_samples(setting_of_generation,number_of_generating_samples=1000):

    desired_sum_of_components = 1 # 合計を指定する特徴量がある場合の、合計の値。例えば、この値を 100 にすれば、合計を 100 にできます
    # 0 から 1 の間の一様乱数でサンプル生成
    np.random.seed(11) # 乱数を生成するためのシードを固定
    x_generated = np.random.rand(number_of_generating_samples, setting_of_generation.shape[1])
    
    # 上限・下限の設定
    x_upper = setting_of_generation.iloc[0, :]  # 上限値
    x_lower = setting_of_generation.iloc[1, :]  # 下限値
    x_generated = x_generated * (x_upper.values - x_lower.values) + x_lower.values  # 上限値から下限値までの間に変換
    
    # 合計を desired_sum_of_components にする特徴量がある場合
    if setting_of_generation.iloc[2, :].sum() != 0:
        for group_number in range(1, int(setting_of_generation.iloc[2, :].max()) + 1):
            variable_numbers = np.where(setting_of_generation.iloc[2, :] == group_number)[0]
            actual_sum_of_components = x_generated[:, variable_numbers].sum(axis=1)
            actual_sum_of_components_converted = np.matlib.repmat(np.reshape(actual_sum_of_components, (x_generated.shape[0], 1)) , 1, len(variable_numbers))
            x_generated[:, variable_numbers] = x_generated[:, variable_numbers] / actual_sum_of_components_converted * desired_sum_of_components
            deleting_sample_numbers, _ = np.where(x_generated > x_upper.values)
            x_generated = np.delete(x_generated, deleting_sample_numbers, axis=0)
            deleting_sample_numbers, _ = np.where(x_generated < x_lower.values)
            x_generated = np.delete(x_generated, deleting_sample_numbers, axis=0)
    
    # 数値の丸め込みをする場合
    if setting_of_generation.shape[0] >= 4:
        x_generated = x_generated.astype(float)
        for variable_number in range(x_generated.shape[1]):
            x_generated[:, variable_number] = np.round(x_generated[:, variable_number], int(setting_of_generation.iloc[3, variable_number]))
# =============================================================================
#             if setting_of_generation.iloc[3, variable_number] == 0:
#                 x_generated[:, variable_number] =  x_generated[:, variable_number].astype(int)
# =============================================================================
    return x_generated


def generating_samples_2(setting_of_generation,number_of_generating_samples=10000):

    desired_sum_of_components = 1 # 合計を指定する特徴量がある場合の、合計の値。例えば、この値を 100 にすれば、合計を 100 にできます
    # 0 から 1 の間の一様乱数でサンプル生成
    np.random.seed(11) # 乱数を生成するためのシードを固定
    x_generated = np.random.rand(number_of_generating_samples, setting_of_generation.shape[1])
    
    # 上限・下限の設定
    x_upper = setting_of_generation.loc['max', :]  # 上限値
    x_lower = setting_of_generation.loc['min', :]  # 下限値
    x_generated = x_generated * (x_upper.values - x_lower.values) + x_lower.values  # 上限値から下限値までの間に変換
    x_generated = pd.DataFrame(x_generated, columns=setting_of_generation.columns)
    
    # 合計を desired_sum_of_components にする特徴量がある場合
    x_generated_groups = pd.DataFrame()
    if setting_of_generation.iloc[2, :].sum() != 0:
        for group_number in range(1, int(setting_of_generation.iloc[2, :].max()) + 1):
            variable_group = setting_of_generation.loc[:,(setting_of_generation.iloc[2, :] == group_number)].columns.tolist()
            x_generated_group = x_generated.loc[:,variable_group]
            #丸め込み
            for column in variable_group:
                x_generated_group.loc[:, column] = float(setting_of_generation.loc['kizami', column]) * np.round(x_generated_group.loc[:, column] /  float(setting_of_generation.loc['kizami', column]))
            x_generated_group.iloc[:,-1] = desired_sum_of_components - x_generated_group.iloc[:,:-1].sum(axis=1)
            
            x_generated_group =  x_generated_group[x_generated_group[variable_group[-1]] <=  x_upper.loc[variable_group[-1]]]
            x_generated_group =  x_generated_group[x_generated_group[variable_group[-1]] >=  x_lower.loc[variable_group[-1]]]
            
            x_generated_groups = pd.concat([x_generated_groups,x_generated_group],axis=1).dropna()
            
    x_generated_no_group = x_generated.loc[x_generated_groups.index, :].drop(x_generated_groups.columns.tolist(),axis=1)
    #丸めこみ
    for j in x_generated_no_group.columns:
        #x_generated[:, variable_number] = np.round(x_generated[:, variable_number], int(setting_of_generation.iloc[3, variable_number]))
        x_generated_no_group.loc[:, j] = float(setting_of_generation.loc['kizami', j]) * np.round(x_generated_no_group.loc[:, j] /  float(setting_of_generation.loc['kizami', j]))
           
    x_generated_final = pd.concat([x_generated_groups, x_generated_no_group],axis=1).reindex(columns=x_generated.columns)
         
    return x_generated_final

def generating_samples_grid(setting_var_dict): 
    '''
    # settings
    number_of_experiments = 30
    variables = {'variable1': [1, 2, 3, 4, 5],
                 'variable2': [-10, 0, 10, 20],
                 'variable3': [0.2, 0.6, 0.8, 1, 1.2]
                 }
    # you can add 'variable4', 'variable5', ... after 'variable3' as well 
    '''
    # make all possible experiments
    all_experiments = np.array(setting_var_dict[list(setting_var_dict)[0]])
    all_experiments = np.reshape(all_experiments, (all_experiments.shape[0], 1))
    for variable_number in range(1, len(setting_var_dict)):
        grid_seed = setting_var_dict[list(setting_var_dict)[variable_number]]
        grid_seed_tmp = matlib.repmat(grid_seed, all_experiments.shape[0], 1)
        all_experiments = np.c_[matlib.repmat(all_experiments, len(grid_seed), 1),
                                np.reshape(grid_seed_tmp.T, (np.prod(grid_seed_tmp.shape), 1))]
    
    x_generated = pd.DataFrame(all_experiments)
    return x_generated


def D_optimization(x_generated, x_obtained=None, number_of_samples=10):
    
    #一旦リセット
    selected_sample_indexes = None
    
    number_of_random_searches = 20000 # ランダムにサンプルを選択して D 最適基準を計算する繰り返し回数
    
    # 実験条件の候補のインデックスの作成
    all_indexes = list(x_generated.index)
    
    # D 最適基準に基づくサンプル選択
    np.random.seed(10) # 乱数を生成するためのシードを固定
    for random_search_number in range(number_of_random_searches):
        # 1. ランダムに候補を選択
        new_selected_indexes = np.random.choice(all_indexes, number_of_samples)
        new_selected_samples = x_generated.loc[new_selected_indexes, :]
        
        if x_obtained is not None:
            new_selected_samples = pd.concat([x_obtained.loc[:,new_selected_samples.columns],new_selected_samples])
            
            
        # 2. オートスケーリングした後に D 最適基準を計算
        autoscaled_new_selected_samples = (new_selected_samples - new_selected_samples.mean()) / new_selected_samples.std()
        xt_x = np.dot(autoscaled_new_selected_samples.T, autoscaled_new_selected_samples)
        d_optimal_value = np.linalg.det(xt_x) 
        # 3. D 最適基準が前回までの最大値を上回ったら、選択された候補を更新
        if random_search_number == 0:
            best_d_optimal_value = d_optimal_value.copy()
            selected_sample_indexes = new_selected_indexes.copy()
        else:
            if best_d_optimal_value < d_optimal_value:
                best_d_optimal_value = d_optimal_value.copy()
                selected_sample_indexes = new_selected_indexes.copy()
    selected_sample_indexes = list(selected_sample_indexes) # リスト型に変換
    
    # 選択されたサンプル、選択されなかったサンプル
    selected_samples = x_generated.loc[selected_sample_indexes, :]  # 選択されたサンプル
    remaining_indexes = list(set(all_indexes) - set(selected_sample_indexes))  # 選択されなかったサンプルのインデックス
    remaining_samples = x_generated.loc[remaining_indexes, :]  # 選択されなかったサンプル
    
    #print(selected_samples.corr()) # 相関行列の確認
    return selected_samples, best_d_optimal_value




#Title
st.title('Design of Experiment')

#data
w = st.sidebar.file_uploader("ファイルアップロード", type='csv')

if w is not None:  #dataをアップロードしたらスタート
    df = pd.read_csv(w, index_col=0)
    
    st.write('Dataset')
    st.dataframe(df)
      
    #説明変数
    var_list = st.sidebar.multiselect('説明変数',df.columns.tolist(),df.columns.tolist()[2:-4])
    var = df.loc[:,var_list]
    
    #候補の作り方
    method = st.sidebar.radio('実験候補の作り方',('数値範囲からランダムに生成', '数値リストの全組み合わせ'))
    
    if method == '数値範囲からランダムに生成':
        settings = pd.DataFrame(index=['upper','lower','group'],columns=var_list)
        
        #使用する説明変数に関しては範囲
        st.sidebar.write('説明変数の範囲')
        for i,variable in enumerate(var.columns):
            with st.sidebar.expander(variable):
                settings.loc['upper',variable] = st.number_input('最大',var[variable].min(),var[variable].max(),var[variable].max())
                settings.loc['lower',variable] = st.number_input('最小',var[variable].min(),var[variable].max(),var[variable].min())
                settings.loc['group',variable] = st.number_input('グループ',0,i+1,0)
        
        number_of_generating_samples = 1000
        var_generated = generating_samples(settings, number_of_generating_samples)
        var_generated = pd.DataFrame(var_generated)
        var_generated.columns = var_list
        var_generated = var_generated.rename(index=lambda s: 'c_'+str(s))
        var_generated = var_generated.astype('float64')       
        #var_generated['End_ratio'] = 2 * var_generated['End_Mn'] /(2 * var_generated['End_Mn'] + var_generated['Mid_Mn']) * 100
    
    elif method == '数値リストの全組み合わせ':
        st.sidebar.write('説明変数のリスト')
        settings = {}
        #使用する説明変数それぞれの値候補リスト
        for i,variable in enumerate(var_list):
            with st.sidebar.expander(variable):
                #settings[variable] = st.text_input(variable+'のリスト', [])
                settings[variable] = st.multiselect('リスト',list(df[variable].unique()),list(df[variable].unique())[:5])

        var_generated = generating_samples_grid(settings)
        var_generated = pd.DataFrame(var_generated)
        var_generated.columns = var_list
        var_generated = var_generated.rename(index=lambda s: 'c_'+str(s))
        var_generated = var_generated.astype('float64')      
        
    
    #main
    st.write('Using data')
    st.dataframe(var)
    #st.write('settings')
    #st.dataframe(settings)
    st.write('生成候補')
    st.dataframe(var_generated)
    
    #D最適計画
    number_of_selecting_samples = st.number_input('D最適基準で選択するサンプル数',1,30,10)
    including_obtained_data = st.checkbox('実験済のデータを考慮するか',value=True)
    if including_obtained_data:
        st.write('実験データも入れてD最適')
        D_selected_samples, d_value = D_optimization(var_generated, x_obtained=var, number_of_samples=number_of_selecting_samples)
    else:
        st.write('実験データなしor0からD最適')
        D_selected_samples, d_value = D_optimization(var_generated, x_obtained=None, number_of_samples=number_of_selecting_samples)
    
    st.dataframe(D_selected_samples)
    st.write(d_value)
    


#     #描画する説明変数
#     st.write('描画する説明変数')
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         X_variable = st.selectbox('説明変数1',var_list)
#     with col2:
#         Y_variable = st.selectbox('説明変数2',var_list,index=1)
#     with col3:
#         Z_variable = st.selectbox('説明変数3',var_list,index=2)
    
#     #2次元散布図
#     plt.rcParams['font.size'] = 6
#     fig, ax = plt.subplots(1,2,figsize=(4, 1.7))

#     ax[0].scatter(df[X_variable],df[Y_variable],s=5, c='blue')
#     ax[0].scatter(D_selected_samples[X_variable],D_selected_samples[Y_variable],s=5, c='red')
#     ax[0].set_xlabel(X_variable,fontsize=6)
#     ax[0].set_ylabel(Y_variable,fontsize=6)
    
#     ax[1].scatter(df[X_variable],df[Z_variable],s=5, c='blue')
#     ax[1].scatter(D_selected_samples[X_variable],D_selected_samples[Z_variable],s=5, c='red')
#     ax[1].set_xlabel(X_variable,fontsize=6)
#     ax[1].set_ylabel(Z_variable,fontsize=6)
#     plt.tight_layout()
#     st.pyplot(fig)
    
#     st.write('実験データ：青、実験候補：赤')
    
    
#     #3次元散布図
#     plt.rcParams['font.size'] = 8
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     # X,Y,Z軸にラベルを設定
#     ax.set_xlabel(X_variable)
#     ax.set_ylabel(Y_variable)
#     ax.set_zlabel(Z_variable)
        
#     ax.plot(df[X_variable],df[Y_variable],df[Z_variable],c='blue',marker="o",linestyle='None')  #実験データ
#     ax.plot(D_selected_samples[X_variable],D_selected_samples[Y_variable],D_selected_samples[Z_variable],c='red',marker="o",linestyle='None') #実験候補   
#     st.pyplot(fig)
    

    
