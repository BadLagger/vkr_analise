import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cross_decomposition import CCA
from scipy import stats
import factor_analyzer.factor_analyzer as fa
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
import warnings
import time
warnings.filterwarnings('ignore')

# ===============================
# 2.4.1 Блок подготовки данных (расширенный)
# ===============================
class DataPreparation:
    """
    Преобразование сырых логов в структурированный датафрейм.
    Автоматическое определение типов, обработка NaN, приведение единиц измерения.
    """

    # Словарь правил приведения на уровне класса
    UNIT_RULES = {
        # Ток
        'current': {'divisor': 1000, 'unit': 'mA_to_A', 'target_unit': 'A'},
        'charger_current': {'divisor': 1000, 'unit': 'mA_to_A', 'target_unit': 'A'},
        'fg_current': {'divisor': 1000, 'unit': 'mA_to_A', 'target_unit': 'A'},
        
        # Напряжение
        'voltage': {'divisor': 1000000, 'unit': 'uV_to_V', 'target_unit': 'V'},
        'charger_voltage': {'divisor': 1000000, 'unit': 'uV_to_V', 'target_unit': 'V'},
        'fg_voltage': {'divisor': 1000000, 'unit': 'uV_to_V', 'target_unit': 'V'},
        
        # Температура (0.1°C -> °C)
        'temp': {'divisor': 10, 'unit': '0.1C_to_C', 'target_unit': '°C'},
        'charger_temp': {'divisor': 10, 'unit': '0.1C_to_C', 'target_unit': '°C'},
        'fg_temp': {'divisor': 10, 'unit': '0.1C_to_C', 'target_unit': '°C'},
        'mcu_temp': {'divisor': 10, 'unit': '0.1C_to_C', 'target_unit': '°C'},
        'temp_sensor': {'divisor': 10, 'unit': '0.1C_to_C', 'target_unit': '°C'},
        
        # Частота (кГц -> Гц)
        'freq': {'multiplier': 1000, 'unit': 'kHz_to_Hz', 'target_unit': 'Hz'},
        'mcu_freq': {'multiplier': 1000, 'unit': 'kHz_to_Hz', 'target_unit': 'Hz'},
    }

    @classmethod
    def get_normalization_rules(cls, column_names):
        """
        Получить словарь с правилами нормализации для указанных колонок.
        
        Args:
            column_names: список имен колонок (с суффиксом _normalized или без)
            
        Returns:
            dict: словарь вида {имя_колонки_без_суффикса: правило_нормализации}
            
        Example:
            >>> rules = DataPreparation.get_normalization_rules(
            ...     ['fg_current_normalized', 'fg_voltage_normalized', 'mcu_temp_normalized']
            ... )
            >>> print(rules)
            {
                'fg_current': {'divisor': 1000, 'unit': 'mA_to_A', 'target_unit': 'A'},
                'fg_voltage': {'divisor': 1000000, 'unit': 'uV_to_V', 'target_unit': 'V'},
                'mcu_temp': {'divisor': 10, 'unit': '0.1C_to_C', 'target_unit': '°C'}
            }
        """
        rules = {}
        
        for col in column_names:
            # Убираем суффикс _normalized, если он есть
            base_col = col.replace('_normalized', '')
            
            # Ищем правило для базового имени колонки
            rule = None
            for key, rule_candidate in cls.UNIT_RULES.items():
                if key == base_col or key in base_col:
                    rule = rule_candidate
                    break
            
            if rule:
                rules[base_col] = rule.copy()
            else:
                # Если правило не найдено, сохраняем информацию, что нормализация не требуется
                rules[base_col] = {'normalized': False, 'message': 'No normalization rule found'}
                print(f"Предупреждение: правило не найдено для колонки '{base_col}'")
        
        return rules

    @staticmethod
    def load_logs(file_path, delimiter=',', convert_units=True):
        """
        Загружает CSV лог в DataFrame.
        convert_units: попытка привести все числовые поля к сопоставимым единицам
        """
        df = pd.read_csv(file_path, delimiter=delimiter)
        
        # Преобразуем timestamp из миллисекунд в секунды (если нужно)
        if 'timestamp' in df.columns:
            # Определяем порядок timestamp (миллисекунды или секунды)
            if df['timestamp'].max() > 1e12:  # больше 1 триллиона -> миллисекунды
                df['timestamp_sec'] = df['timestamp'] / 1000.0
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            else:
                df['timestamp_sec'] = df['timestamp']
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Приведение единиц измерения
        if convert_units:
            df = DataPreparation._normalize_units(df)
        
        return df
    
    @classmethod
    def _normalize_units(cls, df):
        """
        Приведение различных величин к стандартным единицам:
        - ток: мкА -> А (делим на 1_000_000) или мА -> А (делим на 1000)
        - напряжение: мкВ -> В (делим на 1_000_000)
        - температура: 0.1°C -> °C (делим на 10) или 0.001°C -> °C
        - частота: кГц -> Гц (умножаем на 1000) или МГц -> Гц
        """
        df_converted = df.copy()
        
        for col in df.columns:
            if col not in df_converted.columns:
                continue
                
            # Пропускаем нечисловые колонки
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            # Ищем правило для колонки
            rule = None
            for key, rule_candidate in cls.UNIT_RULES.items():
                if key in col.lower():
                    rule = rule_candidate
                    break
            
            if rule:
                if 'divisor' in rule:
                    df_converted[col + '_normalized'] = df[col] / rule['divisor']
                    df_converted[col + '_unit'] = 'A' if 'A' in rule['unit'] else 'V' if 'V' in rule['unit'] else 'C'
                elif 'multiplier' in rule:
                    df_converted[col + '_normalized'] = df[col] * rule['multiplier']
                    df_converted[col + '_unit'] = 'Hz'
                
                # Оставляем оригинал на всякий случай
                print(f"Приведение единиц: {col} -> {col}_normalized ({rule.get('unit', 'unknown')})")
            else:
                # Если правило не найдено, оставляем как есть
                df_converted[col + '_normalized'] = df[col]
        
        return df_converted
    
    @staticmethod
    def handle_missing_values(df, strategy='drop'):
        """
        Обработка пропусков (NaN)
        strategy: 'drop' - удалить строки с NaN,
                  'ffill' - forward fill,
                  'mean' - заполнить средним,
                  'median' - заполнить медианой,
                  'interpolate' - интерполяция
        """
        df_clean = df.copy()
        
        # Считаем количество NaN
        nan_counts = df_clean.isnull().sum()
        if nan_counts.sum() > 0:
            print(f"Обнаружено NaN: {nan_counts[nan_counts > 0].to_dict()}")
        
        if strategy == 'drop':
            df_clean = df_clean.dropna()
            print(f"Удалено строк с NaN: {len(df) - len(df_clean)}")
        elif strategy == 'ffill':
            df_clean = df_clean.fillna(method='ffill')
        elif strategy == 'mean':
            for col in df_clean.select_dtypes(include=[np.number]).columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        elif strategy == 'median':
            for col in df_clean.select_dtypes(include=[np.number]).columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        elif strategy == 'interpolate':
            df_clean = df_clean.interpolate(method='linear')
        
        return df_clean
    
    @staticmethod
    def encode_categorical(df, method='onehot', encoded_cols=None):
        """
        Преобразование категориальных переменных в числовые
        method: 'onehot' - one-hot encoding,
                'label' - label encoding
        """
        df_encoded = df.copy()

        df_cols = df.columns if encoded_cols == None else encoded_cols
        
        for col in df_cols:
            if df[col].dtype == 'object' or (df[col].dtype.name == 'category'):
                if method == 'onehot':
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                    df_encoded = df_encoded.drop(columns=[col])
                elif method == 'label':
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    df_encoded[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                    df_encoded = df_encoded.drop(columns=[col])
        
        return df_encoded

# ===============================
# 2.4.2 Блок анализа данных (адаптированный)
# ===============================
class DataAnalyzer:
    def __init__(self, df, exclude_cols=None):
        """
        exclude_cols: список колонок для исключения из анализа
        (например, временные метки, оригинальные ненормализованные колонки)
        """
        self.df_original = df.copy()
        self.df_processed = df.copy()
        
        # Исключаем явно указанные колонки
        if exclude_cols is None:
            exclude_cols = ['timestamp', 'timestamp_sec', 'datetime', 
                           'charger_status']  # категориальные по умолчанию
        
        # Выбираем числовые колонки для анализа (предпочтительно нормализованные)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Исключаем временные метки и служебные колонки
        self.numerical_cols = [c for c in numeric_cols 
                               if c not in exclude_cols 
                               and not c.endswith('_unit')  # исключаем колонки с единицами
                               and not c.endswith('_original')]  # исключаем оригиналы
        
        print(f"Колонки для анализа: {self.numerical_cols}")
        self.last_scaler = None
        
        # Автоматически определяем масштабы
        self.scales = {}
        for col in self.numerical_cols:
            if col in df.columns:
                self.scales[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'range': df[col].max() - df[col].min()
                }
    
    def get_original_clean(self):
        return self.df_original[self.numerical_cols]
    
    def get_last_scaler(self):
        return self.last_scaler
    
    def get_pca_with_correlations(self, variance_threshold=0.90, n_components=None):
        """
        Выполняет PCA, возвращает датафрейм с исходными параметрами и компонентами,
        а также матрицу корреляций компонент с исходными признаками.
        
        Параметры:
        -----------
        variance_threshold : float
            Доля объясненной дисперсии (по умолчанию 0.90 = 90%)
        n_components : int or None
            Явное число компонент (если None - определяется по threshold)
        
        Возвращает:
        -----------
        dict: {
            'df_with_pca': pd.DataFrame - исходные данные + компоненты,
            'correlations': pd.DataFrame - корреляции компонент с признаками,
            'high_correlations': dict - для каждой компоненты топ-5 признаков,
            'pca': PCA object,
            'explained_variance': array,
            'n_components': int
        }
        """
        # Получаем чистые нормализованные данные
        data = self.get_clean_normalized_dataframe()
        
        if len(data) == 0:
            print("Нет данных для PCA")
            return None
        
        # Копируем исходные данные (ненормализованные, для интерпретации)
        df_original_features = self.df_original[self.numerical_cols].copy()
        
        # Масштабируем для PCA
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Определяем число компонент
        pca_temp = PCA(n_components=n_components)
        pca_temp.fit(data_scaled)
        
        if n_components is None:
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= variance_threshold) + 1
            print(f"Для объяснения {variance_threshold*100:.0f}% дисперсии нужно {n_components} компонент")
        
        # Выполняем PCA с выбранным числом компонент
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(data_scaled)
        
        # Создаем датафрейм с PCA компонентами
        pca_columns = [f'PC{i+1}' for i in range(n_components)]
        df_pca = pd.DataFrame(scores, columns=pca_columns, index=data.index)
        
        # Добавляем исходные параметры (в исходном масштабе, для интерпретации)
        df_with_pca = df_original_features.copy()
        df_with_pca = pd.concat([df_with_pca, df_pca], axis=1)
        
        # Вычисляем корреляции PCA компонент с исходными признаками
        correlations = pd.DataFrame(index=data.columns, columns=pca_columns)
        for col in data.columns:
            for i, pc in enumerate(pca_columns):
                correlations.loc[col, pc] = np.corrcoef(data[col], scores[:, i])[0, 1]
        
        correlations = correlations.astype(float)
        
        # Для каждой компоненты находим топ-5 коррелирующих признаков
        high_correlations = {}
        for pc in pca_columns:
            # Берем абсолютные значения корреляций
            abs_corr = correlations[pc].abs().sort_values(ascending=False)
            top_5 = abs_corr.head(5)
            high_correlations[pc] = {
                'features': top_5.index.tolist(),
                'correlations': correlations.loc[top_5.index, pc].tolist(),
                'abs_correlations': top_5.values.tolist()
            }
        
        # Выводим информацию
        print("\n" + "="*60)
        print("РЕЗУЛЬТАТЫ PCA С ИНТЕРПРЕТАЦИЕЙ")
        print("="*60)
        
        print(f"\nОбъясненная дисперсия по компонентам:")
        for i, ev in enumerate(pca.explained_variance_ratio_):
            print(f"  PC{i+1}: {ev*100:.2f}% (накоплено: {np.sum(pca.explained_variance_ratio_[:i+1])*100:.2f}%)")
        
        print(f"\nСуммарная объясненная дисперсия: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
        
        print("\n" + "-"*60)
        print("ИНТЕРПРЕТАЦИЯ КОМПОНЕНТ (топ-5 коррелирующих признаков):")
        print("-"*60)
        
        for pc in pca_columns:
            print(f"\n{pc} (доля дисперсии: {pca.explained_variance_ratio_[int(pc[2:])-1]*100:.2f}%):")
            for feat, corr, abs_corr in zip(high_correlations[pc]['features'],
                                            high_correlations[pc]['correlations'],
                                            high_correlations[pc]['abs_correlations']):
                sign = "+" if corr > 0 else "-"
                print(f"    {sign} {feat}: {abs_corr:.4f} (r = {corr:.4f})")
        
        # Визуализация
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # График объясненной дисперсии
        axes[0].bar(range(1, n_components+1), pca.explained_variance_ratio_, alpha=0.6, label='Individual')
        axes[0].step(range(1, n_components+1), np.cumsum(pca.explained_variance_ratio_), 
                    where='mid', label='Cumulative', linewidth=2, color='red')
        axes[0].axhline(y=variance_threshold, color='green', linestyle='--', label=f'{variance_threshold*100:.0f}% threshold')
        axes[0].set_xlabel('Principal Components')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('PCA: Explained Variance')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Тепловая карта корреляций
        sns.heatmap(correlations, annot=True, cmap='RdBu_r', center=0, 
                    fmt='.2f', ax=axes[1], cbar_kws={'label': 'Correlation'})
        axes[1].set_title(f'Correlations: Original Features vs PCA Components\n(captures {np.sum(pca.explained_variance_ratio_)*100:.1f}% of variance)')
        axes[1].set_ylabel('Original Features')
        axes[1].set_xlabel('Principal Components')
        
        plt.tight_layout()
        plt.show()
        
        # Дополнительный график: топ-признаки для каждой компоненты
        self._plot_top_features_for_pcs(correlations, pca.explained_variance_ratio_, pca_columns)
        
        return {
            'df_with_pca': df_with_pca,
            'correlations': correlations,
            'high_correlations': high_correlations,
            'pca': pca,
            'scaler': scaler,
            'explained_variance': pca.explained_variance_ratio_,
            'n_components': n_components,
            'scores': scores
        }

    def _plot_top_features_for_pcs(self, correlations, explained_variance, pca_columns, top_n=8):
        """Вспомогательный метод: горизонтальные барчарты топ-признаков для каждой PC"""
        n_pcs = len(pca_columns)
        fig, axes = plt.subplots(1, n_pcs, figsize=(5*n_pcs, 6))
        
        if n_pcs == 1:
            axes = [axes]
        
        for i, pc in enumerate(pca_columns):
            # Сортируем по абсолютной корреляции
            sorted_corr = correlations[pc].abs().sort_values(ascending=False).head(top_n)
            top_features = sorted_corr.index
            top_corrs = correlations.loc[top_features, pc]
            
            # Отрисовка
            colors = ['red' if c < 0 else 'green' for c in top_corrs]
            axes[i].barh(range(len(top_features)), top_corrs.values, color=colors, alpha=0.7)
            axes[i].set_yticks(range(len(top_features)))
            axes[i].set_yticklabels(top_features)
            axes[i].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            axes[i].set_xlabel('Correlation')
            axes[i].set_title(f'{pc}\n({explained_variance[i]*100:.1f}% variance)')
            axes[i].grid(True, alpha=0.3, axis='x')
        
        plt.suptitle(f'Top-{top_n} Features Correlated with Each Principal Component', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def print_data_scale_info(self):
        """Вывод информации о масштабах данных"""
        print("\n=== Информация о масштабах данных ===")
        scale_df = pd.DataFrame(self.scales).T
        print(scale_df.round(2))
        
        # Рекомендации по нормализации
        high_range_cols = scale_df[scale_df['range'] / scale_df['std'] > 100].index.tolist()
        if high_range_cols:
            print(f"\nКолонки с очень большим разбросом: {high_range_cols}")
            print("Рекомендуется использовать RobustScaler или MinMaxScaler")
    
    # 1. Оценка качества данных
    def detect_outliers(self, method='iqr', threshold=1.5):
        """Обнаружение выбросов"""
        outliers_mask = pd.DataFrame(index=self.df_processed.index, 
                                     columns=self.numerical_cols)
        
        for col in self.numerical_cols:
            if col not in self.df_processed.columns:
                continue
            data = self.df_processed[col].dropna()
            if len(data) == 0:
                continue
                
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                outliers_mask[col] = (self.df_processed[col] < lower) | (self.df_processed[col] > upper)
            elif method == 'zscore':
                z = np.abs(stats.zscore(data, nan_policy='omit'))
                threshold_z = threshold if threshold < 5 else 3
                outliers_mask[col] = self.df_processed[col].apply(
                    lambda x: abs(x - data.mean()) / data.std() > threshold_z if pd.notnull(x) else False
                )
        
        return outliers_mask
    
    # 2. Обработка выбросов (с учётом разных масштабов)
    def handle_outliers(self, method='robust_cap', outlier_threshold=1.5):
        """
        method: 'remove', 'cap', 'winsorize', 'robust_cap'
        robust_cap: ограничение на основе медианы и MAD (устойчиво к масштабам)
        """
        if method == 'remove':
            outliers = self.detect_outliers(threshold=outlier_threshold)
            rows_to_drop = outliers.any(axis=1)
            self.df_processed = self.df_processed[~rows_to_drop]
            print(f"Удалено строк с выбросами: {rows_to_drop.sum()}")
        
        elif method == 'cap':
            for col in self.numerical_cols:
                if col not in self.df_processed.columns:
                    continue
                data = self.df_processed[col].dropna()
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - outlier_threshold * IQR
                upper = Q3 + outlier_threshold * IQR
                self.df_processed[col] = self.df_processed[col].clip(lower, upper)
        
        elif method == 'robust_cap':
            for col in self.numerical_cols:
                if col not in self.df_processed.columns:
                    continue
                median = self.df_processed[col].median()
                mad = np.median(np.abs(self.df_processed[col] - median))
                # Ограничиваем в пределах median ± 5*MAD (устойчиво к выбросам)
                lower = median - 5 * mad
                upper = median + 5 * mad
                self.df_processed[col] = self.df_processed[col].clip(lower, upper)
        
        return self.df_processed
    
    def diagnose_data_quality(self):
        """Диагностика качества данных перед анализом"""
        print("\n=== ДИАГНОСТИКА ДАННЫХ ===")
        
        data = self.df_processed[self.numerical_cols]
        
        # 1. Проверка на NaN
        nan_counts = data.isnull().sum()
        if nan_counts.sum() > 0:
            print(f"NaN в данных: {nan_counts[nan_counts > 0].to_dict()}")
        else:
            print("NaN: отсутствуют ✓")
        
        # 2. Проверка на константные колонки
        constant_cols = []
        for col in data.columns:
            if data[col].std() < 1e-10 or data[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            print(f"Константные колонки (будут исключены): {constant_cols}")
        else:
            print("Константные колонки: отсутствуют ✓")
        
        # 3. Проверка на вырожденность корреляционной матрицы
        valid_cols = [c for c in data.columns if c not in constant_cols]
        if len(valid_cols) >= 2:
            corr = data[valid_cols].corr()
            # Проверка на сингулярность
            try:
                eigenvals = np.linalg.eigvalsh(corr.values)
                min_eigenval = eigenvals.min()
                condition = eigenvals.max() / eigenvals.min() if eigenvals.min() > 0 else np.inf
                
                print(f"Минимальное собственное число корреляционной матрицы: {min_eigenval:.6f}")
                print(f"Число обусловленности: {condition:.2f}")
                
                if min_eigenval < 1e-6:
                    print("⚠️  ВНИМАНИЕ: Корреляционная матрица вырождена!")
                    print("   Возможные причины: мультиколлинеарность, недостаточно данных")
                    
                    # Находим сильно коррелированные пары
                    high_corr = []
                    for i in range(len(corr.columns)):
                        for j in range(i+1, len(corr.columns)):
                            if abs(corr.iloc[i, j]) > 0.95:
                                high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
                    
                    if high_corr:
                        print(f"\n   Сильно коррелированные пары (r > 0.95):")
                        for pair in high_corr:
                            print(f"     {pair[0]} ↔ {pair[1]}: {pair[2]:.4f}")
            except Exception as e:
                print(f"Ошибка при вычислении собственных чисел: {e}")
        
        # 4. Проверка на выбросы
        outlier_counts = self.detect_outliers().sum()
        if outlier_counts.sum() > 0:
            print(f"\nВыбросы обнаружены: {outlier_counts[outlier_counts > 0].to_dict()}")
        else:
            print("Выбросы: не обнаружены ✓")
        
        # 5. Размер данных
        print(f"\nРазмер данных: {data.shape[0]} строк, {data.shape[1]} столбцов")
        
        if data.shape[0] < data.shape[1]:
            print("⚠️  ВНИМАНИЕ: Строк меньше чем столбцов!")
            print("   Факторный анализ может быть нестабильным. Рекомендуется:")
            print("   - Уменьшить число переменных")
            print("   - Увеличить количество данных")
        
        # 6. Рекомендации
        print("\n=== РЕКОМЕНДАЦИИ ===")
        if constant_cols:
            print(f"✓ Удалите константные колонки: {constant_cols}")
        
        if data.shape[0] < 50:
            print("✓ Увеличьте объем данных (рекомендуется >50 наблюдений)")
        
        if data.shape[1] > data.shape[0] / 5:
            print("✓ Уменьшите число переменных или увеличьте количество данных")
        
        if len(valid_cols) >= 2:
            corr_abs_mean = corr.abs().values.mean()
            if corr_abs_mean < 0.1:
                print("✓ Корреляции между переменными слабые - факторный анализ может не выявить структуры")
    
    # 3. Нормализация с выбором метода
    def normalize_data(self, method='robust'):
        """
        method: 'standard' - Z-score (StandardScaler)
                'minmax' - MinMaxScaler
                'robust' - RobustScaler (устойчив к выбросам)
        """
        if method == 'standard':
            scaler = StandardScaler()
            self.df_processed[self.numerical_cols] = scaler.fit_transform(
                self.df_processed[self.numerical_cols]
            )
        elif method == 'minmax':
            scaler = MinMaxScaler()
            self.df_processed[self.numerical_cols] = scaler.fit_transform(
                self.df_processed[self.numerical_cols]
            )
        elif method == 'robust':
            scaler = RobustScaler()
            self.df_processed[self.numerical_cols] = scaler.fit_transform(
                self.df_processed[self.numerical_cols]
            )
        self.last_scaler = scaler
        print(f"Нормализация выполнена методом: {method}")
        return self.df_processed
    
    def restore_column(self, column_name, column_data):
        original_df = self.get_original_clean()
        original_temp_idx = list(original_df.columns).index(column_name)
        orig_scale = self.get_last_scaler().scale_[original_temp_idx]
        orig_center = self.get_last_scaler().center_[original_temp_idx]
        return (column_data * orig_scale + orig_center)
    
    def get_clean_normalized_dataframe(self, drop_original=True, drop_constant=True):
        """
        Возвращает датафрейм только с нормализованными колонками.
        
        Параметры:
        -----------
        drop_original : bool
            Удалять ли оригинальные (ненормализованные) колонки
        drop_constant : bool
            Удалять ли константные колонки (все значения одинаковы)
        
        Returns:
        --------
        pd.DataFrame: Очищенный датафрейм с нормализованными данными
        """
        df_clean = self.df_processed[self.numerical_cols].copy()
        
        # 1. Удаляем оригинальные колонки (не нормализованные)
        if drop_original:
            # Колонки без _normalized, но не служебные
            original_cols = [col for col in df_clean.columns 
                            if not col.endswith('_normalized') 
                            and col not in ['datetime', 'timestamp', 'timestamp_sec', 'source_file']
                            and col not in self.numerical_cols]  # оставляем только нормализованные
            
            # Дополнительно: удаляем колонки, у которых есть нормализованная версия
            normalized_cols = [col for col in df_clean.columns if col.endswith('_normalized')]
            for norm_col in normalized_cols:
                original_name = norm_col.replace('_normalized', '')
                if original_name in df_clean.columns:
                    df_clean = df_clean.drop(columns=[original_name])
            
            print(f"Удалено оригинальных колонок: {len([c for c in df_clean.columns if not c.endswith('_normalized')])}")
        
        # 2. Удаляем константные колонки (все значения одинаковы)
        if drop_constant:
            constant_cols = []
            for col in df_clean.columns:
                if df_clean[col].nunique() <= 1:
                    constant_cols.append(col)
            
            if constant_cols:
                df_clean = df_clean.drop(columns=constant_cols)
                print(f"Удалено константных колонок: {constant_cols}")
        
        # 3. Оставляем только числовые колонки
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean = df_clean[numeric_cols]
        
        print(f"\nИтоговый датафрейм: {df_clean.shape[0]} строк, {df_clean.shape[1]} колонок")
        print(f"Колонки: {list(df_clean.columns)}")
        
        return df_clean
    
    # 4. Разделение выборок
    def split_groups(self, group_col=None, n_groups=2, ratios=None):
        """Разделение на группы"""
        if group_col and group_col in self.df_processed.columns:
            groups = [group for _, group in self.df_processed.groupby(group_col)]
        else:
            if ratios is None:
                ratios = [1.0/n_groups] * n_groups
            indices = np.random.permutation(self.df_processed.index)
            split_points = np.cumsum([int(len(indices) * r) for r in ratios[:-1]])
            groups = [self.df_processed.loc[indices[start:end]] 
                      for start, end in zip([0] + split_points.tolist(), 
                                           split_points.tolist() + [len(indices)])]
        return groups
    
    # 5. PCA (адаптирован)
    def perform_pca(self, n_components=None, variance_threshold=0.95):
        """Метод главных компонент"""
        # Чистые нормализированные данные
        data = self.get_clean_normalized_dataframe()
        
        if len(data) == 0:
            print("Нет данных для PCA после удаления NaN")
            return None
        
        # Масштабируем для PCA (обязательно)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        pca = PCA(n_components=n_components)
        pca.fit(data_scaled)
        
        if n_components is None:
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= variance_threshold) + 1
            pca = PCA(n_components=n_components)
            pca.fit(data_scaled)
        
        scores = pca.transform(data_scaled)
        loadings = pca.components_.T
        
        print(f"\n--- PCA Результаты ---")
        print(f"Количество компонент: {n_components}")
        print(f"Доля объясненной дисперсии: {pca.explained_variance_ratio_}")
        print(f"Суммарная доля: {np.sum(pca.explained_variance_ratio_):.3f}")
        
        # Визуализация
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # График дисперсии
        axes[0].bar(range(1, n_components+1), pca.explained_variance_ratio_, alpha=0.6, label='Individual')
        axes[0].step(range(1, n_components+1), np.cumsum(pca.explained_variance_ratio_), 
                     where='mid', label='Cumulative')
        axes[0].set_xlabel('Principal Components')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('PCA: Explained Variance')
        axes[0].legend()
        axes[0].grid()
        
        # Тепловая карта нагрузок
        sns.heatmap(loadings, annot=True, cmap='coolwarm', ax=axes[1],
                    xticklabels=[f'PC{i+1}' for i in range(n_components)],
                    yticklabels=data.columns)
        axes[1].set_title('PCA Loadings')
        
        plt.tight_layout()
        plt.show()
        
        return {'scores': scores, 'loadings': loadings, 'pca': pca, 'scaler': scaler}
    
    # Исправленная версия perform_factor_analysis
    def perform_factor_analysis(self, n_factors=None, rotation='varimax'):

        # Чистые нормализированные данные
        data = self.get_clean_normalized_dataframe()
        """Факторный анализ с проверкой на NaN/Inf"""
        
        # Проверка на достаточное количество данных
        if len(data) < 3:
            print("Недостаточно данных для факторного анализа")
            return None
        
        # Удаляем колонки с нулевой дисперсией (все значения одинаковы)
        constant_cols = []
        for col in data.columns:
            if data[col].std() < 1e-10 or data[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            print(f"Удалены колонки с постоянными значениями: {constant_cols}")
            data = data.drop(columns=constant_cols)
        
        if len(data.columns) < 2:
            print("Недостаточно переменных после удаления константных колонок")
            return None
        
        # Проверка на корректность данных
        if data.isnull().any().any():
            print("Обнаружены NaN в данных, удаляем...")
            data = data.dropna()
        
        # Проверка на бесконечности
        if np.isinf(data.values).any():
            print("Обнаружены inf в данных, заменяем на NaN и удаляем...")
            data = data.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Вычисляем корреляционную матрицу с проверкой
        corr_mtx = data.corr()
        
        # Проверяем корреляционную матрицу на NaN
        if corr_mtx.isnull().any().any():
            print("В корреляционной матрице есть NaN:")
            print(corr_mtx[corr_mtx.isnull().any(axis=1)])
            
            # Находим колонки, вызывающие проблемы
            bad_cols = corr_mtx.columns[corr_mtx.isnull().any(axis=0)].tolist()
            print(f"Проблемные колонки: {bad_cols}")
            
            # Удаляем проблемные колонки
            data = data.drop(columns=bad_cols)
            if len(data.columns) < 2:
                print("После удаления проблемных колонок данных недостаточно")
                return None
            
            corr_mtx = data.corr()
        
        # Проверяем на нулевую или почти нулевую матрицу
        if np.abs(corr_mtx).sum().sum() < 1e-6:
            print("Корреляционная матрица почти нулевая, факторный анализ не имеет смысла")
            return None
        
        # KMO тест с обработкой ошибок
        try:
            kmo_all, kmo_model = calculate_kmo(data)
            print(f"KMO тест: {kmo_model:.3f} (>=0.6 - приемлемо)")
        except Exception as e:
            print(f"KMO тест не удался: {e}")
            kmo_model = 0.5
        
        # Тест сферичности Бартлетта
        try:
            chi2, p_value = calculate_bartlett_sphericity(data)
            print(f"Тест сферичности Бартлетта: chi2={chi2:.1f}, p={p_value:.5f}")
        except Exception as e:
            print(f"Тест Бартлетта не удался: {e}")
            p_value = 1.0
        
        if kmo_model < 0.5 and p_value > 0.05:
            print("Внимание: данные плохо подходят для факторного анализа")
            print("Рекомендуется: увеличить количество данных или проверить корреляции")
        
        # Определение числа факторов
        if n_factors is None:
            try:
                max_factors = min(data.shape[1], data.shape[0] - 1, 10)
                if max_factors < 1:
                    print("Недостаточно данных для определения числа факторов")
                    return None
                
                fa_model_temp = fa.FactorAnalyzer(n_factors=max_factors, rotation=None)
                fa_model_temp.fit(data)
                ev, _ = fa_model_temp.get_eigenvalues()
                n_factors = np.sum(ev > 1)
                print(f"Автоматически выбрано факторов (собственное значение >1): {n_factors}")
            except Exception as e:
                print(f"Автоматическое определение числа факторов не удалось: {e}")
                n_factors = min(2, data.shape[1])
        
        # Проверка на допустимость числа факторов
        max_possible_factors = min(data.shape[1], data.shape[0] - 1)
        if n_factors > max_possible_factors:
            print(f"Число факторов ({n_factors}) превышает максимально возможное ({max_possible_factors})")
            n_factors = max(1, max_possible_factors // 2)
        
        if n_factors < 1:
            print("Некорректное число факторов, использую 1")
            n_factors = 1
        
        # Выполнение факторного анализа
        try:
            fa_model = fa.FactorAnalyzer(n_factors=n_factors, rotation=rotation)
            fa_model.fit(data)
            
            loadings = fa_model.loadings_
            communalities = fa_model.get_communalities()
            
            print(f"\nРезультаты факторного анализа:")
            print(f"Число факторов: {n_factors}")
            print(f"Общности (communalities):")
            for col, comm in zip(data.columns, communalities):
                print(f"  {col}: {comm:.4f}")
            
            # Визуализация нагрузок
            plt.figure(figsize=(12, 6))
            sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0,
                        xticklabels=[f'Factor{i+1}' for i in range(n_factors)],
                        yticklabels=data.columns)
            plt.title(f'Factor Analysis Loadings ({rotation} rotation)')
            plt.tight_layout()
            plt.show()
            
            return {
                'loadings': loadings, 
                'communalities': communalities, 
                'n_factors': n_factors,
                'variables': data.columns.tolist(),
                'model': fa_model
            }
        
        except Exception as e:
            print(f"Ошибка при выполнении факторного анализа: {e}")
            print("Попробуйте уменьшить число факторов или использовать другой rotation")
            
            # Fallback с меньшим числом факторов
            try:
                n_factors_fallback = max(1, n_factors - 1)
                print(f"Повторная попытка с {n_factors_fallback} факторами...")
                fa_model = fa.FactorAnalyzer(n_factors=n_factors_fallback, rotation=rotation)
                fa_model.fit(data)
                return {
                    'loadings': fa_model.loadings_,
                    'communalities': fa_model.get_communalities(),
                    'n_factors': n_factors_fallback,
                    'variables': data.columns.tolist(),
                    'model': fa_model
                }
            except:
                print("Факторный анализ не удался даже с меньшим числом факторов")
                return None
    
    # 7. Каноническая корреляция
    def perform_cca(self, set1_cols, set2_cols, n_components=2):

        # Чистые нормализированные данные
        data = self.get_clean_normalized_dataframe()
        
        # Проверяем наличие колонок
        set1_cols = [c for c in set1_cols if c in data.columns]
        set2_cols = [c for c in set2_cols if c in data.columns]
        
        if len(set1_cols) == 0 or len(set2_cols) == 0:
            print("Нет доступных колонок для CCA")
            return None
        
        X = data[set1_cols]
        Y = data[set2_cols]
        
        # Масштабирование
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        Y_scaled = scaler.fit_transform(Y)
        
        n_comp = min(n_components, X.shape[1], Y.shape[1])
        cca = CCA(n_components=n_comp)
        cca.fit(X_scaled, Y_scaled)
        
        X_c, Y_c = cca.transform(X_scaled, Y_scaled)
        correlations = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_comp)]
        
        print(f"\n--- Каноническая корреляция ---")
        print(f"Канонические корреляции: {correlations}")
        print(f"Корреляции (в квадрате): {[c**2 for c in correlations]}")
        
        # Визуализация
        fig, axes = plt.subplots(1, min(2, n_comp), figsize=(12, 5))
        if n_comp == 1:
            axes = [axes]
        
        for i in range(min(2, n_comp)):
            axes[i].scatter(X_c[:, i], Y_c[:, i], alpha=0.6)
            axes[i].set_xlabel(f'Canonical variate {i+1} (X)')
            axes[i].set_ylabel(f'Canonical variate {i+1} (Y)')
            axes[i].set_title(f'CCA pair {i+1}, corr={correlations[i]:.3f}')
            axes[i].grid()
        
        plt.tight_layout()
        plt.show()
        
        return {'x_scores': X_c, 'y_scores': Y_c, 'correlations': correlations, 'cca': cca}
    
    # 8. Выделение значимых параметров
    def extract_important_features(self, method='pca', n_components=2, threshold=0.6):
        data = self.get_clean_normalized_dataframe()
        """Выделение наиболее значимых параметров"""
        if method == 'pca':
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            pca = PCA(n_components=min(n_components, len(data.columns)))
            pca.fit(data_scaled)
            loadings = np.abs(pca.components_.T)
            
            important = set()
            for i in range(min(n_components, loadings.shape[1])):
                for j, col in enumerate(data.columns):
                    if j < len(loadings) and loadings[j, i] > threshold:
                        important.add(col)
        
        elif method == 'fa':
            n_facts = min(n_components, len(data.columns))
            fa_model = fa.FactorAnalyzer(n_factors=n_facts, rotation='varimax')
            fa_model.fit(data)
            loadings = np.abs(fa_model.loadings_)
            
            important = set()
            for i in range(n_facts):
                for j, col in enumerate(data.columns):
                    if loadings[j, i] > threshold:
                        important.add(col)
        else:
            raise ValueError("method must be 'pca' or 'fa'")
        
        return list(important)
# ===============================
# Подготовка Данных
# ===============================  
def data_prepare(df_orig):
    print("=== Исходные данные ===")
    print(df_orig.head())
    print("\nТипы данных:")
    print(df_orig.dtypes)
    
    # Блок подготовки данных
    print("\n" + "="*50)
    print("Блок подготовки данных")
    print("="*50)
    
    # 1. Приведение единиц измерения
    df_prepared = DataPreparation._normalize_units(df)
    
    # 2. Кодирование категориальных переменных
    df_prepared = DataPreparation.encode_categorical(df_prepared, method='label')
    
    # 3. Обработка возможных NaN (в данном примере их нет)
    df_prepared = DataPreparation.handle_missing_values(df_prepared, strategy='drop')
    
    print("\nДанные после подготовки:")
    print(df_prepared.head())
    
    # Блок анализа данных
    print("\n" + "="*50)
    print("Блок анализа данных")
    print("="*50)
    return df_prepared
# ===============================
# Анализ Данных
# ===============================    
def analyze(df_orig):
        # Исключаем служебные колонки из анализа
    exclude = ['timestamp', 
               'charger_current', 
               'charger_temp', 
               'charger_voltage',
               'display_brightness', 
               'fg_capacity', 
               'fg_current', 
               'fg_temp',
               'fg_voltage', 
               'mcu_freq', 
               'mcu_temp', 
               'second_core_state',
               'temp_sensor',
               'timestamp_normalized',
               'charger_current_unit_encoded',
               'charger_temp_unit_encoded', 
               'charger_voltage_unit_encoded',
               'fg_current_unit_encoded', 
               'fg_temp_unit_encoded',
               'fg_voltage_unit_encoded', 
               'mcu_freq_unit_encoded',
               'mcu_temp_unit_encoded', 
               'temp_sensor_unit_encoded']
    
    analyzer = DataAnalyzer(df_orig, exclude_cols=exclude)
    
    # Информация о масштабах
    analyzer.print_data_scale_info()

    
    # Обнаружение выбросов
    print("\n=== Обнаружение выбросов ===")
    outliers = analyzer.detect_outliers(method='iqr')
    print(f"Количество выбросов по колонкам:\n{outliers.sum()}")
    
    # Обработка выбросов (устойчивый метод)
    analyzer.handle_outliers(method='robust_cap')
    
    # Нормализация (RobustScaler - хорош для данных с разными масштабами)
    analyzer.normalize_data(method='robust')

    # Диагностика качества данных
    analyzer.diagnose_data_quality()
    
    # PCA
    pca_result = analyzer.perform_pca(n_components=8)
    
    # Факторный анализ
    fa_result = analyzer.perform_factor_analysis(n_factors=5)
    if fa_result is None:
        print("Факторный анализ не удался. Возможные решения:")
        print("1. Проверьте корреляционную матрицу")
        print("2. Уменьшите число факторов")
        print("3. Исключите некоторые переменные вручную")
        
        # Альтернатива: попробуем с 1 фактором
        print("\nПопытка с 1 фактором...")
        fa_result = analyzer.perform_factor_analysis(n_factors=1)
    
    # Каноническая корреляция (разделим на температурные и электрические параметры)
    temp_cols = [c for c in analyzer.numerical_cols if 'temp' in c.lower()]
    electrical_cols = [c for c in analyzer.numerical_cols if 'current' in c.lower() or 'voltage' in c.lower()]
    
    if len(temp_cols) >= 1 and len(electrical_cols) >= 1:
        cca_result = analyzer.perform_cca(temp_cols, electrical_cols, n_components=1)
    
    # Выделение значимых параметров
    important_pca = analyzer.extract_important_features(method='pca', n_components=3, threshold=0.46)
    print(f"\n=== Значимые параметры (PCA, threshold=0.45): {important_pca}")

    # ============================================
    # PCA с интерпретацией через корреляции
    # ============================================
    print("\n" + "="*60)
    print("PCA анализ с интерпретацией компонент")
    print("="*60)

    # Получаем PCA компоненты и их корреляции с исходными признаками
    pca_interpretation = analyzer.get_pca_with_correlations(variance_threshold=0.95)

    if pca_interpretation is not None:
        # Доступ к датафрейму с PCA компонентами
        df_with_pca = pca_interpretation['df_with_pca']
        print(f"\nДатафрейм с исходными параметрами и PCA компонентами:")
        print(df_with_pca.head())
        
        # Сохраняем для дальнейшего использования
        # df_with_pca.to_csv('data_with_pca_components.csv', index=False)
        
        # Доступ к корреляциям
        correlations = pca_interpretation['correlations']
        print(f"\nМатрица корреляций компонент с признаками:")
        print(correlations.round(3))
        
        # Доступ к интерпретации (для каждой PC - какие признаки важны)
        high_corrs = pca_interpretation['high_correlations']
        print(f"\nДетальная интерпретация:")
        for pc, info in high_corrs.items():
            print(f"\n{pc}:")
            for feat, corr in zip(info['features'], info['correlations']):
                print(f"    {feat}: {corr:.3f}")
        
        # ============================================
        # Теперь можно строить модель на основе интерпретированных факторов
        # ============================================
        print("\n" + "="*60)
        print("ФОРМИРОВАНИЕ ФАКТОРОВ ДЛЯ МОДЕЛИ")
        print("="*60)
        
        # На основе анализа корреляций определяем скрытые факторы
        factors_for_model = {}
        
        for pc, info in high_corrs.items():
            # Берем признаки с корреляцией > 0.5 или топ-3
            strong_features = []
            for feat, corr in zip(info['features'], info['correlations']):
                if abs(corr) > 0.5:
                    strong_features.append(feat)
            
            # Даем интерпретируемое имя фактору
            if 'temp' in ' '.join(strong_features).lower():
                factor_name = f"Factor_Temperature_{pc}"
            elif 'current' in ' '.join(strong_features).lower():
                factor_name = f"Factor_Power_{pc}"
            elif 'voltage' in ' '.join(strong_features).lower():
                factor_name = f"Factor_Voltage_{pc}"
            else:
                factor_name = f"Factor_{pc}"
            
            factors_for_model[factor_name] = {
                'pc_column': pc,
                'features': strong_features,
                'correlations': [info['correlations'][i] for i in range(len(strong_features))]
            }
            
            print(f"\n{factor_name}:")
            print(f"  Состав: {strong_features}")
            print(f"  Корреляции: {[round(c, 3) for c in factors_for_model[factor_name]['correlations']]}")
        
        # Теперь df_with_pca содержит PC1, PC2, PC3, PC4, которые можно использовать
        # как признаки в любой модели (регрессия, классификация, кластеризация)
        
        # Пример: подготовка данных для модели
        feature_columns = [f'PC{i+1}' for i in range(pca_interpretation['n_components'])]
        X_for_model = df_with_pca[feature_columns]  # Факторы для модели
        
        # Если у вас есть целевая переменная (target), то:
        # y_target = df_with_pca['some_target_column']
        # И дальше строить модель: model.fit(X_for_model, y_target)
        
        print(f"\nГотово! Получено {len(feature_columns)} факторов для моделирования:")
        print(f"  Факторы: {feature_columns}")
        print(f"  Размер матрицы признаков: {X_for_model.shape}")

# ===============================
# Создание модели на PCA
# ===============================
# ===============================
# Создание модели на PCA
# ===============================
def pca_modeling(df_orig):
    """
    Построение модели предсказания charger_temp_normalized на основе PCA компонент
    """
    print("\n" + "="*60)
    print("МОДЕЛИРОВАНИЕ НА ОСНОВЕ PCA КОМПОНЕНТ")
    print("="*60)
    
    # Создаем анализатор (как в analyze функции)
    exclude = ['timestamp', 
               'charger_current', 
               'charger_temp', 
               'charger_voltage',
               'display_brightness', 
               'fg_capacity', 
               'fg_current', 
               'fg_temp',
               'fg_voltage', 
               'mcu_freq', 
               'mcu_temp', 
               'second_core_state',
               'temp_sensor',
               'timestamp_normalized',
               'charger_current_normalized',
               'charger_voltage_normalized',
               'mcu_freq_normalized',
               'second_core_state_normalized',
               'charger_status_encoded',
               'charger_current_unit_encoded',
               'charger_temp_unit_encoded', 
               'charger_voltage_unit_encoded',
               'fg_current_unit_encoded', 
               'fg_temp_unit_encoded',
               'fg_voltage_unit_encoded', 
               'mcu_freq_unit_encoded',
               'mcu_temp_unit_encoded', 
               'temp_sensor_unit_encoded']
    
    analyzer = DataAnalyzer(df_orig, exclude_cols=exclude)
    
    # Обработка выбросов и нормализация
    analyzer.handle_outliers(method='robust_cap')
    analyzer.normalize_data(method='robust')
    
    # Получаем чистые данные для PCA
    clean_data = analyzer.get_clean_normalized_dataframe()
    #print("Данные после PCA")
    #print(clean_data)

    #restore_temp = analyzer.restore_column('charger_temp_normalized', clean_data['charger_temp_normalized'])

    #print(restore_temp)
    #sys.exit(0)    
    # Проверяем наличие целевой переменной
    target_col = 'charger_temp_normalized'
    if target_col not in clean_data.columns:
        print(f"Ошибка: целевая переменная '{target_col}' не найдена в данных")
        print(f"Доступные колонки: {list(clean_data.columns)}")
        return None
    
    # 1. Исключаем целевую переменную из признаков
    feature_cols = [col for col in clean_data.columns if col != target_col]
    X = clean_data[feature_cols].copy()
    y = clean_data[target_col].copy()
    
    print(f"\nИсходные данные:")
    print(f"  - Количество признаков: {X.shape[1]}")
    print(f"  - Количество наблюдений: {X.shape[0]}")
    print(f"  - Целевая переменная: {target_col}")
    
    # 2. Применяем PCA для получения 3 компонент
    print("\n" + "-"*40)
    print("Выполнение PCA (3 компоненты)...")
    print("-"*40)
    
    start_scale_time = time.time()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scale_time = time.time() - start_scale_time
    
    pca_start_time = time.time()
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    # Создаем датафрейм с PCA компонентами
    pca_columns = [f'PC{i+1}' for i in range(3)]
    df_pca = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
    pca_time = time.time() - pca_start_time
    
    # 3. Присоединяем целевую переменную
    df_model = df_pca.copy()
    df_model[target_col] = y.values
    
    print(f"\nРезультат PCA:")
    print(f"  - Объясненная дисперсия по компонентам: {pca.explained_variance_ratio_}")
    print(f"  - Суммарная объясненная дисперсия: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
    print(f"  - Форма датафрейма для модели: {df_model.shape}")
    print(f"  - Колонки: {list(df_model.columns)}")
    print(f"  - Время масштабирования (мс): {scale_time*1000}")
    print(f"  - Время масштабирования семпла (мс): {(scale_time / len(X)) * 1000}")
    print(f"  - Время создания PCA (мс): {pca_time*1000}")
    print(f"  - Время создания PCA (мс) семпла: {(pca_time/len(X_scaled))*1000}")
    
    # Визуализация объясненной дисперсии
    plt.figure(figsize=(10, 4))
    plt.bar(range(1, 4), pca.explained_variance_ratio_, alpha=0.6, label='Individual')
    plt.plot(range(1, 4), np.cumsum(pca.explained_variance_ratio_), 
             'ro-', label='Cumulative', linewidth=2)
    plt.axhline(y=0.95, color='green', linestyle='--', label='95% threshold')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA: Explained Variance (4 components)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 4))
    plt.tight_layout()
    plt.show()
    
    # Покажем нагрузки (feature importance для PCA)
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=pca_columns,
        index=feature_cols
    )
    print("\nНагрузки (loadings) компонент на исходные признаки:")
    print(loadings_df.round(3))
    
    # 4. Разделение на обучающую и тестовую выборки (80/20)
    from sklearn.model_selection import train_test_split
    
    X_model = df_model[pca_columns]
    y_model = df_model[target_col]
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_model, y_model, 
        test_size=0.2, 
        random_state=42,
        shuffle=True
    )
    
    print(f"\nРазделение выборок:")
    print(f"  - Обучающая выборка: {X_train.shape[0]} строк")
    print(f"  - Тестовая выборка: {X_test.shape[0]} строк")
    
    # 5. Обучение моделей
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    }
    
    results = {}
    
    for model_name, model in models.items():
        print("\n" + "="*50)
        print(f"Обучение модели: {model_name}")
        print("="*50)
        
        # Обучение
        train_start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_start_time
        print(f"  - Время тренировки (с): {train_time}")
        print(f"  - Время тренировки сэмпла (мс): {(train_time/len(X_train)) * 1000}")
        
        # Предсказания
        train_start_time = time.time()
        y_train_pred = model.predict(X_train)
        y_train_pred_restored = analyzer.restore_column('charger_temp_normalized', y_train_pred)
        train_time = time.time() - train_start_time
        print(f"  - Время предсказания на обучающей выборке (с): {train_time}")
        print(f"  - Время предсказания сэмпла на обучающей выборке (мс): {(train_time/len(X_train)) * 1000}")
        print(f"  - Восстановленная температура: min={min(y_train_pred_restored)}, max={max(y_train_pred_restored)}")

        train_start_time = time.time()
        y_test_pred = model.predict(X_test)
        y_test_pred_restored = analyzer.restore_column('charger_temp_normalized', y_test_pred)
        train_time = time.time() - train_start_time
        print(f"  - Время предсказания на тестовой выборке (с): {train_time}")
        print(f"  - Время предсказания сэмпла на тестовой выборке (мс): {(train_time/len(X_test)) * 1000}")
        print(f"  - Восстановленная температура: min={min(y_test_pred_restored)}, max={max(y_test_pred_restored)}")

        # Метрики на обучающей выборке
        y_train_restored = analyzer.restore_column('charger_temp_normalized', y_train)
        train_r2 = r2_score(y_train_restored, y_train_pred_restored)
        train_mae = mean_absolute_error(y_train_restored, y_train_pred_restored)
        train_rmse = np.sqrt(mean_squared_error(y_train_restored, y_train_pred_restored))
        
        # Метрики на тестовой выборке
        y_test_restored = analyzer.restore_column('charger_temp_normalized', y_test)
        test_r2 = r2_score(y_test_restored, y_test_pred_restored)
        test_mae = mean_absolute_error(y_test_restored, y_test_pred_restored)
        test_rmse = np.sqrt(mean_squared_error(y_test_restored, y_test_pred_restored))
        
        # Сохраняем результаты
        results[model_name] = {
            'model': model,
            'train': {'r2': train_r2, 'mae': train_mae, 'rmse': train_rmse},
            'test': {'r2': test_r2, 'mae': test_mae, 'rmse': test_rmse},
            'predictions': y_test_pred_restored,
            'y_true': y_test_restored
        }
        
        # Выводим результаты
        print(f"\nРезультаты на обучающей выборке:")
        print(f"  - R² = {train_r2:.4f}")
        print(f"  - MAE = {train_mae:.4f}")
        print(f"  - RMSE = {train_rmse:.4f}")
        
        print(f"\nРезультаты на тестовой выборке:")
        print(f"  - R² = {test_r2:.4f}")
        print(f"  - MAE = {test_mae:.4f}")
        print(f"  - RMSE = {test_rmse:.4f}")
        
        # Проверка на переобучение
        if train_r2 - test_r2 > 0.1:
            print(f"\n⚠️  ВНИМАНИЕ: Возможно переобучение!")
            print(f"   Разница R²: {train_r2 - test_r2:.4f}")
    
    # 6. Визуализация результатов
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for idx, (model_name, result) in enumerate(results.items()):
        row = idx // 2
        col = idx % 2
        
        # График предсказанных vs реальных значений
        axes[row, col].scatter(result['y_true'], result['predictions'], 
                              alpha=0.5, edgecolors='k', linewidth=0.5)
        
        # Линия идеального предсказания
        min_val = min(result['y_true'].min(), result['predictions'].min())
        max_val = max(result['y_true'].max(), result['predictions'].max())
        axes[row, col].plot([min_val, max_val], [min_val, max_val], 
                           'r--', linewidth=2, label='Ideal')
        
        axes[row, col].set_xlabel('Actual Temperature')
        axes[row, col].set_ylabel('Predicted Temperature')
        axes[row, col].set_title(f'{model_name}\nR² = {result["test"]["r2"]:.4f}, MAE = {result["test"]["mae"]:.4f}')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    # Удаляем пустые подграфики (если моделей меньше 4)
    if len(results) < 4:
        for idx in range(len(results), 4):
            row = idx // 2
            col = idx % 2
            axes[row, col].axis('off')
    
    plt.suptitle('Сравнение моделей предсказания температуры зарядки', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 7. Сравнительная диаграмма метрик
    metrics_df = pd.DataFrame({
        model_name: {
            'R² (test)': result['test']['r2'],
            'MAE (test)': result['test']['mae'],
            'RMSE (test)': result['test']['rmse']
        }
        for model_name, result in results.items()
    }).T
    
    print("\n" + "="*50)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*50)
    print(metrics_df.round(4))
    
    # Визуализация сравнения метрик
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_df.plot(kind='bar', ax=ax, rot=0)
    ax.set_title('Сравнение метрик моделей на тестовой выборке')
    ax.set_xlabel('Модель')
    ax.set_ylabel('Значение метрики')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    # 8. Важность компонент (на основе Random Forest)
    best_model_name = max(results, key=lambda x: results[x]['test']['r2'])
    best_model = results[best_model_name]['model']
    
    if hasattr(best_model, 'feature_importances_'):
        print(f"\nАнализ важности PCA компонент (на основе {best_model_name}):")
        importances = best_model.feature_importances_
        for name, imp in zip(pca_columns, importances):
            print(f"  {name}: {imp:.4f}")
        
        # Визуализация важности компонент
        plt.figure(figsize=(8, 5))
        plt.bar(pca_columns, importances)
        plt.xlabel('PCA Components')
        plt.ylabel('Feature Importance')
        plt.title(f'Важность компонент в модели {best_model_name}')
        plt.tight_layout()
        plt.show()
    
    # 9. Возвращаем результаты для дальнейшего использования
    return {
        'results': results,
        'pca': pca,
        'scaler': scaler,
        'df_model': df_model,
        'pca_components': pca_columns,
        'best_model': best_model_name,
        'metrics': metrics_df
    }


# ===============================
# Моделирование на исходных признаках (без PCA)
# ===============================
def clean_modeling(df_prepared):
    """
    Построение модели предсказания charger_temp_normalized на основе исходных признаков
    Без использования PCA, только выбранные признаки
    """
    print("\n" + "="*60)
    print("МОДЕЛИРОВАНИЕ НА ИСХОДНЫХ ПРИЗНАКАХ (БЕЗ PCA)")
    print("="*60)
    
    # Целевая переменная
    target_col = 'charger_temp_normalized'
    
    # Выбранные признаки для моделирования
    feature_cols = [
        #'display_brightness_normalized',
        #'fg_capacity_normalized', 
        'fg_current_normalized', 
        'fg_voltage_normalized', 
        'mcu_temp_normalized'
    ]
    
    # Проверяем наличие всех колонок
    available_features = []
    missing_features = []
    
    for col in feature_cols:
        if col in df_prepared.columns:
            available_features.append(col)
        else:
            missing_features.append(col)
    
    if missing_features:
        print(f"\n⚠️  ВНИМАНИЕ: Отсутствуют колонки: {missing_features}")
        print(f"   Используем только доступные: {available_features}")
    
    if target_col not in df_prepared.columns:
        print(f"\n❌ ОШИБКА: Целевая переменная '{target_col}' не найдена")
        print(f"   Доступные колонки: {list(df_prepared.columns)}")
        return None
    
    # Формируем признаковое пространство и целевую переменную
    X = df_prepared[available_features].copy()
    y = df_prepared[target_col].copy()
    
    print(f"\nИсходные данные:")
    print(f"  - Количество признаков: {X.shape[1]}")
    print(f"  - Признаки: {available_features}")
    print(f"  - Количество наблюдений: {X.shape[0]}")
    print(f"  - Целевая переменная: {target_col}")
    
    # Статистика по данным
    print("\nСтатистика признаков:")
    print(X.describe().round(4))
    
    # Проверка корреляций с целевой переменной
    print("\n" + "-"*40)
    print("АНАЛИЗ КОРРЕЛЯЦИЙ С ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")
    print("-"*40)
    
    correlations = {}
    for col in available_features:
        corr = np.corrcoef(X[col], y)[0, 1]
        correlations[col] = corr
        print(f"  {col}: {corr:.4f}")
    
    # Визуализация корреляций
    plt.figure(figsize=(10, 6))
    corr_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Correlation'])
    corr_df = corr_df.sort_values('Correlation', ascending=False)
    
    colors = ['red' if c < 0 else 'green' for c in corr_df['Correlation']]
    plt.barh(corr_df['Feature'], corr_df['Correlation'], color=colors, alpha=0.7)
    plt.xlabel('Correlation with charger_temp_normalized')
    plt.title('Корреляция признаков с целевой переменной')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    
    # 4. Разделение на обучающую и тестовую выборки (80/20)
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        shuffle=True
    )
    
    print(f"\nРазделение выборок:")
    print(f"  - Обучающая выборка: {X_train.shape[0]} строк")
    print(f"  - Тестовая выборка: {X_test.shape[0]} строк")
    
    # ============================================
    # ОБУЧЕНИЕ МОДЕЛЕЙ
    # ============================================
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        ),
        'Linear Regression': LinearRegression()
    }
    
    results = {}
    
    for model_name, model in models.items():
        print("\n" + "="*50)
        print(f"Обучение модели: {model_name}")
        print("="*50)
        
        # Обучение (разные методы для разных типов моделей)

        train_start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_start_time
        print(f"  - Время обучения (с): {train_time}")
        print(f"  - Время обучения сэмпла (мс): {(train_time/len(X_train)) * 1000}")

        train_start_time = time.time()
        y_train_pred = model.predict(X_train)
        train_time = time.time() - train_start_time
        print(f"  - Время предсказания на обучающей выборке (с): {train_time}")
        print(f"  - Время предскзания сэмпла на обучающей выборке (мс): {(train_time/len(X_train)) * 1000}")

        train_start_time = time.time()
        y_test_pred = model.predict(X_test)
        train_time = time.time() - train_start_time
        print(f"  - Время предсказания на тестовой выборке (с): {train_time}")
        print(f"  - Время предскзания сэмпла на тестовой выборке (мс): {(train_time/len(X_test)) * 1000}")
        
        # Метрики на обучающей выборке
        train_r2 = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        
        # Метрики на тестовой выборке
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Сохраняем результаты
        results[model_name] = {
            'model': model,
            'train': {'r2': train_r2, 'mae': train_mae, 'rmse': train_rmse},
            'test': {'r2': test_r2, 'mae': test_mae, 'rmse': test_rmse},
            'predictions': y_test_pred,
            'y_true': y_test
        }
        
        # Выводим результаты
        print(f"\nРезультаты на обучающей выборке:")
        print(f"  - R² = {train_r2:.4f}")
        print(f"  - MAE = {train_mae:.4f}")
        print(f"  - RMSE = {train_rmse:.4f}")
        
        print(f"\nРезультаты на тестовой выборке:")
        print(f"  - R² = {test_r2:.4f}")
        print(f"  - MAE = {test_mae:.4f}")
        print(f"  - RMSE = {test_rmse:.4f}")
        
        # Проверка на переобучение
        if train_r2 - test_r2 > 0.05:
            print(f"\n⚠️  ВНИМАНИЕ: Возможно переобучение!")
            print(f"   Разница R²: {train_r2 - test_r2:.4f}")
    
    # 6. Визуализация результатов
    n_models = len(results)
    n_cols = 2
    n_rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (model_name, result) in enumerate(results.items()):
        row = idx // n_cols
        col = idx % n_cols
        
        # График предсказанных vs реальных значений
        axes[row, col].scatter(result['y_true'], result['predictions'], 
                              alpha=0.5, edgecolors='k', linewidth=0.5, s=10)
        
        # Линия идеального предсказания
        min_val = min(result['y_true'].min(), result['predictions'].min())
        max_val = max(result['y_true'].max(), result['predictions'].max())
        axes[row, col].plot([min_val, max_val], [min_val, max_val], 
                           'r--', linewidth=2, label='Ideal')
        
        axes[row, col].set_xlabel('Actual Temperature')
        axes[row, col].set_ylabel('Predicted Temperature')
        axes[row, col].set_title(f'{model_name}\nR² = {result["test"]["r2"]:.4f}, MAE = {result["test"]["mae"]:.4f}')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    # Удаляем пустые подграфики
    for idx in range(len(results), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('Сравнение моделей предсказания температуры зарядки', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 7. Сравнительная диаграмма метрик
    metrics_df = pd.DataFrame({
        model_name: {
            'R² (test)': result['test']['r2'],
            'MAE (test)': result['test']['mae'],
            'RMSE (test)': result['test']['rmse']
        }
        for model_name, result in results.items()
    }).T
    
    print("\n" + "="*50)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*50)
    print(metrics_df.round(4))
    
    # 8. Визуализация сравнения метрик
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_df.plot(kind='bar', ax=ax, rot=15)
    ax.set_title('Сравнение метрик моделей на тестовой выборке')
    ax.set_xlabel('Модель')
    ax.set_ylabel('Значение метрики')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    # 9. Важность признаков (только для моделей, поддерживающих это)
    print("\n" + "="*50)
    print("ВАЖНОСТЬ ПРИЗНАКОВ / АНАЛИЗ ВЕСОВ")
    print("="*50)
    
    for model_name, result in results.items():
        model = result['model']
        if hasattr(model, 'feature_importances_'):
            print(f"\n{model_name} - важность признаков:")
            importances = model.feature_importances_
            for feat, imp in zip(available_features, importances):
                print(f"  {feat}: {imp:.4f}")
            
            # Визуализация важности
            plt.figure(figsize=(10, 6))
            importance_df = pd.DataFrame({
                'Feature': available_features,
                'Importance': importances
            }).sort_values('Importance', ascending=True)
            
            plt.barh(importance_df['Feature'], importance_df['Importance'])
            plt.xlabel('Feature Importance')
            plt.title(f'Важность признаков - {model_name}')
            plt.tight_layout()
            plt.show()
        
        elif hasattr(model, 'weights') and model_name == 'Kalman Regressor (with features)':
            print(f"\n{model_name} - веса признаков:")
            for feat, weight in zip(available_features, model.weights):
                print(f"  {feat}: {weight:.4f}")
            
            # Визуализация весов
            plt.figure(figsize=(10, 6))
            weights_df = pd.DataFrame({
                'Feature': available_features,
                'Weight': model.weights
            }).sort_values('Weight', ascending=True)
            
            colors = ['red' if w < 0 else 'green' for w in weights_df['Weight']]
            plt.barh(weights_df['Feature'], weights_df['Weight'], color=colors)
            plt.xlabel('Weight')
            plt.title(f'Веса признаков - {model_name}')
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            plt.tight_layout()
            plt.show()
    
    # 10. Нахождение лучшей модели
    best_model_name = max(results, key=lambda x: results[x]['test']['r2'])
    best_model = results[best_model_name]['model']
    
    print("\n" + "="*50)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*50)
    print(f"\n✅ Лучшая модель: {best_model_name}")
    print(f"   - R² на тесте: {results[best_model_name]['test']['r2']:.6f}")
    print(f"   - MAE: {results[best_model_name]['test']['mae']:.6f}")
    print(f"   - RMSE: {results[best_model_name]['test']['rmse']:.6f}")
    
    # Сравнение фильтров Калмана
    #print("\n📊 Сравнение фильтров Калмана:")
    #kalman_reg_r2 = results.get('Kalman Regressor (with features)', {}).get('test', {}).get('r2', 0)
    
    #print(f"   Kalman с признаками: R² = {kalman_reg_r2:.6f}")
    
    #if kalman_reg_r2 > kalman_ts_r2:
    #    print(f"   ✓ Добавление признаков улучшило фильтр Калмана на {kalman_reg_r2 - kalman_ts_r2:.6f} (R²)")
    #else:
    #    print(f"   ℹ️  Для временного ряда достаточно истории температуры")
    
    # 11. Предсказание для примера
    print("\n" + "="*50)
    print("ПРИМЕР ПРЕДСКАЗАНИЯ")
    print("="*50)
    
    # Берем первые 5 строк из тестовой выборки
    sample_idx = X_test.index[:5]
    sample_features = X_test.loc[sample_idx]
    sample_actual = y_test.loc[sample_idx]
    
    predictions_df = pd.DataFrame({
        'Actual': sample_actual.values
    })
    
    for model_name, result in results.items():
        predictions_df[f'{model_name}_Pred'] = result['predictions'][:5]
        predictions_df[f'{model_name}_Error'] = sample_actual.values - result['predictions'][:5]
    
    print("\nПример предсказаний для 5 случайных наблюдений:")
    print(predictions_df.round(6))
    
    # 12. Анализ ошибок по диапазонам температуры
    print("\n" + "="*50)
    print("АНАЛИЗ ОШИБОК ПО ДИАПАЗОНАМ ТЕМПЕРАТУРЫ")
    print("="*50)
    
    best_result = results[best_model_name]
    y_true = best_result['y_true']
    y_pred = best_result['predictions']
    errors = np.abs(y_true - y_pred)
    
    # Разбиваем на квантили по температуре
    temp_quantiles = pd.qcut(y_true, q=4, labels=['Низкая', 'Средняя-низкая', 'Средняя-высокая', 'Высокая'])
    
    error_by_temp = pd.DataFrame({
        'Temperature_Range': temp_quantiles,
        'Error': errors
    }).groupby('Temperature_Range')['Error'].agg(['mean', 'std', 'max'])
    
    print(f"\nАнализ ошибок модели {best_model_name} по диапазонам температуры:")
    print(error_by_temp.round(6))
    
    # Дополнительный анализ для фильтров Калмана
    #if 'Kalman Regressor (with features)' in results:
    #    kalman_errors = np.abs(results['Kalman Regressor (with features)']['y_true'] - 
    #                           results['Kalman Regressor (with features)']['predictions'])
    #    print(f"\nСравнение с Kalman Regressor:")
    #    print(f"  Средняя ошибка Kalman: {kalman_errors.mean():.6f}")
    #    print(f"  Средняя ошибка лучшей модели ({best_model_name}): {errors.mean():.6f}")
    #    print(f"  Улучшение: {(kalman_errors.mean() - errors.mean()) / kalman_errors.mean() * 100:.2f}%")
    
    return {
        'results': results,
        'feature_cols': available_features,
        'target_col': target_col,
        'best_model': best_model_name,
        'best_model_object': best_model,
        'metrics': metrics_df,
        'correlations': correlations
    }

def analyze_error_distribution(results, model_name='XGBoost'):
    """
    Детальный анализ распределения ошибок модели
    """
    print("\n" + "="*60)
    print("АНАЛИЗ РАСПРЕДЕЛЕНИЯ ОШИБОК")
    print("="*60)
    
    # Получаем результаты лучшей модели
    result = results[model_name]
    y_true = result['y_true']
    y_pred = result['predictions']
    errors = np.abs(y_true - y_pred)
    relative_errors = (errors / y_true) * 100
    
    # Статистика ошибок
    print(f"\n📊 Статистика абсолютных ошибок (°C):")
    print(f"  - Средняя ошибка (MAE): {errors.mean():.4f}")
    print(f"  - Медианная ошибка: {errors.median():.4f}")
    print(f"  - Стандартное отклонение: {errors.std():.4f}")
    print(f"  - 95-й процентиль: {errors.quantile(0.95):.4f}")
    print(f"  - 99-й процентиль: {errors.quantile(0.99):.4f}")
    print(f"  - Максимальная ошибка: {errors.max():.4f}")
    
    # Анализ по порогам
    thresholds = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0]
    print(f"\n📈 Частота ошибок по порогам:")
    print(f"{'Порог (°C)':<12} {'Кол-во':<10} {'Процент':<10} {'Накопленный %':<15}")
    print("-" * 50)
    
    cumulative = 0
    for thresh in thresholds:
        count = (errors > thresh).sum()
        percentage = (count / len(errors)) * 100
        cumulative += percentage
        print(f"{thresh:<12} {count:<10} {percentage:.4f}%     {cumulative:.4f}%")
    
    # Визуализация распределения ошибок
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Гистограмма ошибок
    axes[0, 0].hist(errors, bins=100, alpha=0.7, edgecolor='black', density=True)
    axes[0, 0].axvline(errors.mean(), color='red', linestyle='--', label=f'Mean: {errors.mean():.3f}')
    axes[0, 0].axvline(errors.median(), color='green', linestyle='--', label=f'Median: {errors.median():.3f}')
    axes[0, 0].axvline(errors.quantile(0.95), color='orange', linestyle='--', label=f'95th: {errors.quantile(0.95):.3f}')
    axes[0, 0].set_xlabel('Absolute Error (°C)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title(f'Distribution of Absolute Errors\n{model_name}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Box plot ошибок
    bp = axes[0, 1].boxplot(errors, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    axes[0, 1].set_ylabel('Absolute Error (°C)')
    axes[0, 1].set_title(f'Box Plot of Errors\n{model_name}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Добавляем статистику на box plot
    stats_text = f"Q1: {errors.quantile(0.25):.3f}\nQ3: {errors.quantile(0.75):.3f}\nIQR: {errors.quantile(0.75) - errors.quantile(0.25):.3f}"
    axes[0, 1].text(0.7, errors.quantile(0.75), stats_text, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    
    # 3. QQ plot (для проверки нормальности)
    from scipy import stats as scipy_stats
    scipy_stats.probplot(errors, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title(f'Q-Q Plot of Errors\n{model_name}')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Зависимость ошибки от реальной температуры
    axes[1, 1].scatter(y_true, errors, alpha=0.3, s=5)
    axes[1, 1].set_xlabel('Actual Temperature (°C)')
    axes[1, 1].set_ylabel('Absolute Error (°C)')
    axes[1, 1].set_title(f'Error vs Actual Temperature\n{model_name}')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Добавляем линию тренда
    z = np.polyfit(y_true, errors, 1)
    p = np.poly1d(z)
    axes[1, 1].plot(y_true.sort_values(), p(y_true.sort_values()), "r--", alpha=0.8, label=f'Trend: {z[0]:.4f}°C/°C')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Анализ больших ошибок
    print(f"\n🔍 Анализ больших ошибок ( > 2°C ):")
    large_errors_mask = errors > 2.0
    large_errors_count = large_errors_mask.sum()
    large_errors_percent = (large_errors_count / len(errors)) * 100
    
    print(f"  - Количество больших ошибок: {large_errors_count} из {len(errors)} ({large_errors_percent:.4f}%)")
    
    if large_errors_count > 0:
        large_errors_df = pd.DataFrame({
            'Actual': y_true[large_errors_mask],
            'Predicted': y_pred[large_errors_mask],
            'Absolute_Error': errors[large_errors_mask],
            'Relative_Error_%': relative_errors[large_errors_mask]
        }).sort_values('Absolute_Error', ascending=False)
        
        print(f"\n  Топ-10 самых больших ошибок:")
        print(large_errors_df.head(10).round(2))
        
        # Анализ температуры, где возникают большие ошибки
        print(f"\n  Температурные диапазоны больших ошибок:")
        temp_ranges = pd.cut(y_true[large_errors_mask], bins=[0, 20, 30, 40, 50, 60, 70, 80])
        range_counts = temp_ranges.value_counts().sort_index()
        for temp_range, count in range_counts.items():
            print(f"    {temp_range}: {count} ошибок ({count/large_errors_count*100:.1f}%)")
    
    # Анализ относительных ошибок
    print(f"\n📊 Относительные ошибки (% от реальной температуры):")
    print(f"  - Средняя относительная ошибка: {relative_errors.mean():.4f}%")
    print(f"  - Медианная относительная ошибка: {relative_errors.median():.4f}%")
    print(f"  - Максимальная относительная ошибка: {relative_errors.max():.4f}%")
    
    # Вероятность ошибки > 1°C, > 2°C, > 5°C
    print(f"\n🎯 Вероятность превышения порога:")
    for threshold in [0.5, 1.0, 2.0, 5.0]:
        prob = (errors > threshold).mean() * 100
        print(f"  P(error > {threshold}°C) = {prob:.4f}%")
    
    return {
        'errors': errors,
        'relative_errors': relative_errors,
        'stats': {
            'mean': errors.mean(),
            'median': errors.median(),
            'std': errors.std(),
            'q95': errors.quantile(0.95),
            'q99': errors.quantile(0.99),
            'max': errors.max()
        },
        'large_errors': large_errors_df if large_errors_count > 0 else None
    }

# Добавьте этот импорт в начало файла
import treelite
#import treelite_runtime
import joblib
import os
from datetime import datetime

# ===============================
# Сравнение моделей и сохранение лучшей
# ===============================
def compare_and_save_best_model(pca_results, clean_results, df_prepared, output_dir='models'):
    """
    Сравнивает модели из PCA и clean моделирования, выбирает лучшую и сохраняет в формате Treelite
    
    Параметры:
    -----------
    pca_results : dict
        Результаты из pca_modeling()
    clean_results : dict
        Результаты из clean_modeling()
    df_prepared : pd.DataFrame
        Подготовленные данные (для восстановления признаков)
    output_dir : str
        Директория для сохранения моделей
    """
    print("\n" + "="*70)
    print("СРАВНЕНИЕ МОДЕЛЕЙ И СОХРАНЕНИЕ ЛУЧШЕЙ")
    print("="*70)
    
    # Создаем директорию для моделей
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Извлекаем лучшие модели из каждого подхода
    models_comparison = {}
    
    # PCA модели
    if pca_results and 'results' in pca_results:
        for model_name, result in pca_results['results'].items():
            model_key = f"PCA_{model_name}"
            models_comparison[model_key] = {
                'model': result['model'],
                'test_r2': result['test']['r2'],
                'test_mae': result['test']['mae'],
                'test_rmse': result['test']['rmse'],
                'approach': 'PCA',
                'features': pca_results['pca_components'],
                'pca': pca_results.get('pca'),
                'scaler': pca_results.get('scaler'),
                'is_pca': True
            }
    
    # Clean модели
    if clean_results and 'results' in clean_results:
        for model_name, result in clean_results['results'].items():
            model_key = f"Clean_{model_name}"
            models_comparison[model_key] = {
                'model': result['model'],
                'test_r2': result['test']['r2'],
                'test_mae': result['test']['mae'],
                'test_rmse': result['test']['rmse'],
                'approach': 'Clean',
                'features': clean_results.get('feature_cols', []),
                'pca': None,
                'scaler': None,
                'is_pca': False
            }
    
    # Создаем DataFrame для сравнения
    comparison_df = pd.DataFrame([
        {
            'Model': name,
            'Approach': info['approach'],
            'R²': info['test_r2'],
            'MAE': info['test_mae'],
            'RMSE': info['test_rmse'],
            'Features': len(info['features'])
        }
        for name, info in models_comparison.items()
    ]).sort_values('R²', ascending=False)
    
    print("\n📊 СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ:")
    print("="*70)
    print(comparison_df.to_string(index=False))
    
    # Визуализация сравнения
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    # R² сравнение
    axes[0].barh(comparison_df['Model'], comparison_df['R²'], 
                 color=['green' if x > 0.9 else 'orange' if x > 0.7 else 'red' 
                        for x in comparison_df['R²']])
    axes[0].set_xlabel('R² Score')
    axes[0].set_title('Comparison of R² Scores')
    axes[0].axvline(x=0.9, color='green', linestyle='--', alpha=0.5, label='Excellent (0.9)')
    axes[0].axvline(x=0.7, color='orange', linestyle='--', alpha=0.5, label='Good (0.7)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # MAE сравнение
    axes[1].barh(comparison_df['Model'], comparison_df['MAE'], color='skyblue')
    axes[1].set_xlabel('MAE (°C)')
    axes[1].set_title('Comparison of MAE')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # RMSE сравнение
    axes[2].barh(comparison_df['Model'], comparison_df['RMSE'], color='lightcoral')
    axes[2].set_xlabel('RMSE (°C)')
    axes[2].set_title('Comparison of RMSE')
    axes[2].grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Model Comparison: PCA vs Clean Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Выбираем лучшую модель
    best_model_name = comparison_df.iloc[0]['Model']
    best_model_info = models_comparison[best_model_name]
    
    print("\n" + "="*70)
    print(f"🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model_name}")
    print("="*70)
    print(f"  - Подход: {best_model_info['approach']}")
    print(f"  - R²: {best_model_info['test_r2']:.6f}")
    print(f"  - MAE: {best_model_info['test_mae']:.6f}")
    print(f"  - RMSE: {best_model_info['test_rmse']:.6f}")
    print(f"  - Количество признаков: {len(best_model_info['features'])}")
    print(f"  - Признаки: {best_model_info['features']}")
    
    # Сохраняем модель в формате Treelite
    print("\n" + "="*70)
    print("СОХРАНЕНИЕ МОДЕЛИ В ФОРМАТЕ TREELITE")
    print("="*70)
    
    best_model = best_model_info['model']
    
    # Проверяем, что модель поддерживает сохранение в Treelite
    if hasattr(best_model, 'get_booster'):
        try:
            # Сохраняем модель в формате Treelite
            model_name = f"temperature_predictor_{timestamp}"
            
            # Получаем booster из модели XGBoost
            booster = best_model.get_booster()
            #print("Тип бустера: %s" % type(booster))
            # Конвертируем в Treelite модель
            tl_model = treelite.frontend.from_xgboost(booster)
            #print("Тип модели: %s" % type(tl_model))
            #print(f"Доступные методы: {[m for m in dir(tl_model) if not m.startswith('_')]}")
            # Сохраняем в файл (.so или .dll)
            treelite_path = os.path.join(output_dir, f"{model_name}.so")

            #tl_model.export_lib(
            #    toolchain='gcc',
            #    libpath=treelite_path,
            #    verbose=True
            #)
            
            model_bytes = tl_model.serialize_bytes()
            with open(treelite_path, 'wb') as f:
                f.write(model_bytes)
            
            print(f"✅ Модель сохранена в Treelite формате: {treelite_path}")
            # Сохраняем метаданные модели
            metadata = {
                'model_name': model_name,
                'approach': best_model_info['approach'],
                'features': best_model_info['features'],
                'test_r2': best_model_info['test_r2'],
                'test_mae': best_model_info['test_mae'],
                'test_rmse': best_model_info['test_rmse'],
                'timestamp': timestamp,
                'feature_count': len(best_model_info['features']),
                'is_pca': best_model_info['is_pca']
            }
            
            # Если это PCA модель, сохраняем PCA и scaler
            if best_model_info['is_pca'] and best_model_info['pca'] is not None:
                pca_path = os.path.join(output_dir, f"{model_name}_pca.pkl")
                scaler_path = os.path.join(output_dir, f"{model_name}_scaler.pkl")
                
                joblib.dump(best_model_info['pca'], pca_path)
                joblib.dump(best_model_info['scaler'], scaler_path)
                
                metadata['pca_path'] = pca_path
                metadata['scaler_path'] = scaler_path
                metadata['pca_components'] = best_model_info['pca'].n_components_
                
                print(f"✅ PCA сохранен: {pca_path}")
                print(f"✅ Scaler сохранен: {scaler_path}")
            
            # Сохраняем метаданные в JSON
            import json
            metadata_path = os.path.join(output_dir, f"{model_name}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Метаданные сохранены: {metadata_path}")
            
            # Создаем пример использования на Go
            #create_go_example(output_dir, model_name, metadata)
            
            # Сохраняем также стандартным способом (на всякий случай)
            joblib_path = os.path.join(output_dir, f"{model_name}_model.pkl")
            joblib.dump(best_model, joblib_path)
            print(f"✅ Модель сохранена в pickle: {joblib_path}")

            # Сохранение параметров нормализации
            normalization_prms = DataPreparation.get_normalization_rules(best_model_info['features'])
            normalization_prms_path=os.path.join(output_dir, f"{model_name}_normalization.json")
            with open(normalization_prms_path, 'w') as f:
                json.dump(normalization_prms, f, indent=2)
            print(f"✅ Параметры нормализации сохранены: {normalization_prms_path}")
            
            # Возвращаем информацию о сохраненной модели
            saved_info = {
                'model_name': model_name,
                'treelite_path': treelite_path,
                'metadata_path': metadata_path,
                'joblib_path': joblib_path,
                'metadata': metadata,
                'model': best_model
            }
            
            return saved_info
            
        except Exception as e:
            print(f"❌ Ошибка при сохранении в Treelite: {e}")
            print("   Сохраняем модель в стандартном формате...")
            
            # Fallback: сохраняем в pickle
            joblib_path = os.path.join(output_dir, f"best_model_{timestamp}.pkl")
            joblib.dump(best_model, joblib_path)
            print(f"✅ Модель сохранена в pickle: {joblib_path}")
            
            return {'joblib_path': joblib_path, 'model': best_model}
    else:
        print("⚠️  Модель не поддерживает сохранение в Treelite (не XGBoost)")
        print("   Сохраняем в стандартном формате pickle...")
        
        joblib_path = os.path.join(output_dir, f"best_model_{timestamp}.pkl")
        joblib.dump(best_model, joblib_path)
        print(f"✅ Модель сохранена в pickle: {joblib_path}")
        
        return {'joblib_path': joblib_path, 'model': best_model}



# ===============================
# Пример использования с реальными данными
# ===============================
if __name__ == "__main__":
    # Создаём DataFrame из предоставленных данных
    #data_str = """timestamp,charger_current,charger_status,charger_temp,charger_voltage,display_brightness,fg_capacity,fg_current,fg_temp,fg_voltage,mcu_freq,mcu_temp,second_core_state,temp_sensor
#1776283323418,465000,Charging,550,4678000,15,51,-155937,464,3819140,300000,60000,0,5567
#1776283324418,465000,Charging,550,4677000,15,51,-155937,464,3819062,300000,60000,0,5567
#1776283325418,465000,Charging,550,4679000,15,51,-156250,463,3819062,300000,60000,0,5567
#1776283326418,465000,Charging,550,4678000,15,51,-156250,463,3819062,300000,60000,0,5567
#1776283327418,465000,Charging,550,4678000,15,51,-155625,463,3819062,300000,60000,0,5566
#1776283328418,465000,Charging,550,4677000,15,51,-155312,463,3819062,300000,60000,0,5566
#1776283329418,465000,Charging,550,4679000,15,51,-155468,463,3819062,300000,60000,0,5566"""
    
    # Загружаем данные
    #from io import StringIO
    #df = pd.read_csv("/home/hrechko/yadisk/Учёба/ВКР/vkr_log/metrics_20260416_030203.csv")
    
    import glob
    import sys

    WIN_PATH="G:/Yadisk/YandexDisk/Учёба/ВКР/vkr_log/metrics_*.csv"
    LIN_PATH="/home/hrechko/yadisk/Учёба/ВКР/vkr_log/metrics_*.csv"
    files = glob.glob(LIN_PATH if sys.platform == "linux" else WIN_PATH)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    #if sys.platform != "linux":
    #    matplotlib.use('Qt5Agg')

    df_prepared = data_prepare(df)

    print(f"Normalized charger temp: min={min(df_prepared['charger_temp_normalized'])}, max={max(df_prepared['charger_temp_normalized'])}")
    #analyze(df_prepared)
    results_pca = pca_modeling(df_prepared)
    best_model_name = max(results_pca['results'], key=lambda x: results_pca['results'][x]['test']['r2'])
    error_analysis = analyze_error_distribution(results_pca['results'], best_model_name)

    results_clean = clean_modeling(df_prepared)

    best_model_name = max(results_clean['results'], key=lambda x: results_clean['results'][x]['test']['r2'])
    error_analysis = analyze_error_distribution(results_clean['results'], best_model_name)

    result = compare_and_save_best_model(results_pca, results_clean, df_prepared)
