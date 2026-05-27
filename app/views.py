from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework import status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from rest_framework.parsers import JSONParser
from rest_framework import status
from django.http import HttpResponse
import csv
from io import StringIO
import numpy as np
from django.conf import settings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from django.http import JsonResponse
from django.views import View
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.signal import butter, filtfilt
import plotly.graph_objects as go
import os
import io
import requests
import base64
from django.core.files.storage import default_storage
from django.views.decorators.csrf import csrf_exempt
# Sklearn library
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from django.core.files.storage import default_storage

# 1. Zero Crossing Rate (ZCR)
def calculate_zcr(signal, frame_length, hop_length):
    zcr_values = []
    for start in range(0, len(signal) - frame_length + 1, hop_length):
        frame = signal[start:start + frame_length]
        zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_length)
        zcr_values.append(zcr)
    return zcr_values

# 2. DASDV (Standard og'ish)
def calculate_dasdv(signal):
    return np.mean(np.abs(np.diff(signal)))

# 3. MAV (O'rtacha mutlaq qiymat)
def calculate_mav(signal):
    return np.mean(np.abs(signal))

# 4. G (Summation)
def calculate_g(signal):
    G = np.sum(signal)  
    return G

# 5. SSI (Sign Signal Integral)
def calculate_ssi(signal):
    return np.sum(np.power(signal, 2))

# 6. VAR (Dispersiya)
def calculate_var(signal):
    return np.var(signal)

# 7. TM3 va TM5 (Temporal Moments)
def calculate_tm3(signal):
    return np.mean(np.power(signal, 3))

def calculate_tm5(signal):
    return np.mean(np.power(signal, 5))

# 8. RMS (Root Mean Square)
def calculate_rms(signal):
    return np.sqrt(np.mean(np.power(signal, 2)))

# 9. LOG (Logarithmic Signal)
def calculate_log(signal):
    return np.sum(np.log(np.abs(signal) + 1))

# 10. WL (Waveform Length)
def calculate_wl(signal):
    return np.sum(np.abs(np.diff(signal)))

# 11. AAC (Average Amplitude Change)
def calculate_aac(signal):
    return np.mean(np.abs(np.diff(signal)))

class MyTokenObtainPairView(TokenObtainPairView):
    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # Tokenlarni olish
        tokens = serializer.validated_data

        # Foydalanuvchi ob'ektini olish
        user = serializer.user
        
        # Foydalanuvchining superuser yoki staff ekanligini tekshirish
        is_superuser = user.is_superuser
        is_staff = user.is_staff

        # Tokenlarga qo'shimcha ma'lumot qo'shish
        tokens['is_superuser'] = is_superuser
        tokens['is_staff'] = is_staff

        return Response(tokens, status=status.HTTP_200_OK)

class ProtectedView(APIView):
    permission_classes = [IsAuthenticated]  # Faqat autentifikatsiyalangan foydalanuvchilar uchun

    def get(self, request):
        # Token to'g'ri bo'lsa, bu ma'lumot qaytariladi
        return Response({"message": "Bu himoyalangan yo'lga kirdingiz!"}, status=200)
class ZCRView(View):
    def get(self, request):
        
        url_path = request.path
        url_path=url_path.split('/')
        print("URL path:", url_path)  # Masalan, "/api/ok/zcr/"


        # CSV faylini o'qish
        data = pd.read_csv(settings.BASE_DIR / f'{url_path[2]}pro.csv')

        # Signal ustunini olish
        signal = data['1'].values

        # Freymlar uzunligini belgilash
        frame_length = 8
        hop_length = frame_length // 2

        # ZCR ni hisoblash
        def calculate_zcr(signal, frame_length, hop_length):
            zcr_values = []
            for start in range(0, len(signal) - frame_length + 1, hop_length):
                frame = signal[start:start + frame_length]
                zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_length)
                zcr_values.append(zcr)
            return zcr_values

        # ZCR ni hisoblash
        zcr_values = calculate_zcr(signal, frame_length, hop_length)

        # Grafikni chizish
        plt.figure(figsize=(10, 6))
        plt.plot(zcr_values)
        plt.title('Zero Crossing Rate')
        plt.xlabel('Frame')
        plt.ylabel('ZCR')

        # Rasmni saqlash
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()  # Grafikni to'xtatish

        # Rasmni base64 formatiga o'girish
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
       
        # Rasmni JSON formatida yuborish
        return JsonResponse({'image': f'data:image/png;base64,{image_base64}'})
    


# Har bir xususiyat uchun umumiy grafik chizish va qaytarish funksiyasi
def generate_graph(signal, feature_name):
    plt.figure(figsize=(12, 6))
    plt.plot(signal, label='Signal')
    plt.title(f'{feature_name} Grafik')
    plt.xlabel('Frame')
    plt.ylabel('Amplitude')
    plt.legend()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# MAV
class MAVView(View):
    def get(self, request):
        url_path = request.path
        url_path=url_path.split('/')
        data = pd.read_csv(settings.BASE_DIR / f'{url_path[2]}pro.csv')
        signal = data['1'].values
        mav_value = calculate_mav(signal)
        image_base64 = generate_graph(signal, 'MAV')
        return JsonResponse({'mav': mav_value, 'image': f'data:image/png;base64,{image_base64}'})

# DASDV
class DASDVView(View):
    def get(self, request):
        url_path = request.path
        url_path=url_path.split('/')
        data = pd.read_csv(settings.BASE_DIR / f'{url_path[2]}pro.csv')
        signal = data['2'].values
        dasdv_value = calculate_dasdv(signal)
        image_base64 = generate_graph(signal, 'DASDV')
        return JsonResponse({'dasdv': dasdv_value, 'image': f'data:image/png;base64,{image_base64}'})
class GView(View):
    def get(self, request):
        
        url_path = request.path
        url_path=url_path.split('/')
        data = pd.read_csv(settings.BASE_DIR / f'{url_path[2]}pro.csv')
        signal = data['3'].values
        g_value = calculate_g(signal)
        image_base64 = generate_graph(signal, 'G')
        return JsonResponse({'g': g_value, 'image': f'data:image/png;base64,{image_base64}'})

# SSI
class SSIView(View):
    def get(self, request):
        url_path = request.path
        url_path=url_path.split('/')
        data = pd.read_csv(settings.BASE_DIR / f'{url_path[2]}pro.csv')
        signal = data['4'].values
        ssi_value = calculate_ssi(signal)
        image_base64 = generate_graph(signal, 'SSI')
        return JsonResponse({'ssi': ssi_value, 'image': f'data:image/png;base64,{image_base64}'})

# VAR
class VARView(View):
    def get(self, request):
        url_path = request.path
        url_path=url_path.split('/')
        data = pd.read_csv(settings.BASE_DIR / f'{url_path[2]}pro.csv')
        signal = data['5'].values
        var_value = calculate_var(signal)
        image_base64 = generate_graph(signal, 'VAR')
        return JsonResponse({'var': var_value, 'image': f'data:image/png;base64,{image_base64}'})

# TM3
class TM3View(View):
    def get(self, request):
        url_path = request.path
        url_path=url_path.split('/')
        data = pd.read_csv(settings.BASE_DIR / f'{url_path[2]}pro.csv')
        signal = data['6'].values
        tm3_value = calculate_tm3(signal)
        image_base64 = generate_graph(signal, 'TM3')
        return JsonResponse({'tm3': tm3_value, 'image': f'data:image/png;base64,{image_base64}'})

# TM5
class TM5View(View):
    def get(self, request):
        url_path = request.path
        url_path=url_path.split('/')
        data = pd.read_csv(settings.BASE_DIR / f'{url_path[2]}pro.csv')
        signal = data['7'].values
        tm5_value = calculate_tm5(signal)
        image_base64 = generate_graph(signal, 'TM5')
        return JsonResponse({'tm5': tm5_value, 'image': f'data:image/png;base64,{image_base64}'})

# RMS
class RMSView(View):
    def get(self, request):
        url_path = request.path
        url_path=url_path.split('/')
        data = pd.read_csv(settings.BASE_DIR / f'{url_path[2]}pro.csv')
        signal = data['1'].values
        rms_value = calculate_rms(signal)
        image_base64 = generate_graph(signal, 'RMS')
        return JsonResponse({'rms': rms_value, 'image': f'data:image/png;base64,{image_base64}'})

# LOG
class LOGView(View):
    def get(self, request):
        url_path = request.path
        url_path=url_path.split('/')
        data = pd.read_csv(settings.BASE_DIR / f'{url_path[2]}pro.csv')
        signal = data['2'].values
        log_value = calculate_log(signal)
        image_base64 = generate_graph(signal, 'LOG')
        return JsonResponse({'log': log_value, 'image': f'data:image/png;base64,{image_base64}'})

# WL (Waveform Length)
class WLView(View):
    def get(self, request):
        url_path = request.path
        url_path=url_path.split('/')
        data = pd.read_csv(settings.BASE_DIR / f'{url_path[2]}pro.csv')
        signal = data['3'].values
        wl_value = calculate_wl(signal)
        image_base64 = generate_graph(signal, 'WL')
        return JsonResponse({'wl': wl_value, 'image': f'data:image/png;base64,{image_base64}'})

# AAC (Average Amplitude Change)
class AACView(View):
    def get(self, request):
        url_path = request.path
        url_path=url_path.split('/')
        data = pd.read_csv(settings.BASE_DIR / f'{url_path[2]}pro.csv')
        signal = data['4'].values
        aac_value = calculate_aac(signal)
        image_base64 = generate_graph(signal, 'AAC')
        return JsonResponse({'aac': aac_value, 'image': f'data:image/png;base64,{image_base64}'})
# myapp/views.py

class Filtering(View):
    def get(self, request):
        url_path = request.path.split('/')
        data = pd.read_csv(settings.BASE_DIR / f'{url_path[2]}pro.csv')

        # Signalni olish
        original_signal = data['1']

        # Filtr yaratish
        def butter_lowpass(cutoff, fs, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return b, a

        def butter_lowpass_filter(data, cutoff, fs, order=5):
            b, a = butter_lowpass(cutoff, fs, order=order)
            y = filtfilt(b, a, data)
            return y

        # Filtr parametrlari
        order = 6
        fs = 30.0  # namunalar olish tezligi
        cutoff = 3.0  # kesish chastotasi

        # Signalni filtrlash
        filtered_signal = butter_lowpass_filter(original_signal, cutoff, fs, order)

        # Grafik yaratish
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(original_signal, label='Original Signal', color='blue')
        ax.plot(filtered_signal, label='Filtered Signal', color='red')
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.set_title('Original and Filtered Signals')
        ax.legend()

        # Grafikni BytesIO obyektiga saqlash
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')

        # Tasvirni JSON javob sifatida qaytarish
        return JsonResponse({'image': image_base64})
class Scaling(View):
    def get(self, request):
        url_path = request.path.split('/')
        data = pd.read_csv(settings.BASE_DIR / f'{url_path[2]}pro.csv')

        # Signalni olish
        original_signal = data['1']

        # Scaling funksiyasi
        def scale_signal(signal):
            min_val = signal.min()
            max_val = signal.max()
            scaled_signal = (signal - min_val) / (max_val - min_val)  # [0, 1] oralig'iga keltirish
            return scaled_signal

        # Signalni masshtablashtirish
        scaled_signal = scale_signal(original_signal)

        # Grafik yaratish
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(original_signal, label='Original Signal', color='blue')
        ax.plot(scaled_signal, label='Scaled Signal', color='green')
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.set_title('Original and Scaled Signals')
        ax.legend()

        # Grafikni BytesIO obyektiga saqlash
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')

        # Tasvirni JSON javob sifatida qaytarish
        return JsonResponse({'image': image_base64})
class FourierTransformView(View):
    def get(self, request):
        # CSV faylini o'qish
        
        url_path = request.path.split('/')
        data = pd.read_csv(settings.BASE_DIR / f'{url_path[2]}pro.csv')

        # Assumption: CSV faylida 'Signal' nomli ustun bor
        if '1' not in data.columns:
            return JsonResponse({'error': 'Signal column not found in the CSV file.'}, status=400)

        # Signalni olish
        signal = data['1'].values
        fs = 1000  # O'lchov chastotasi (Hz)
        t = np.linspace(0, len(signal)/fs, len(signal), endpoint=False)  # Vaqt vektori

        # Fourier Transformini amalga oshirish
        fft_values = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(signal), d=1/fs)

        # Grafikni yaratish
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))

        # 1. Asl signal grafi
        ax[0].plot(t, signal)
        ax[0].set_title('Original Signal')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Amplitude')

        # 2. Fourier Transform grafi
        ax[1].plot(frequencies[:len(frequencies)//2], np.abs(fft_values)[:len(fft_values)//2])
        ax[1].set_title('Fourier Transform')
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_ylabel('Magnitude')

        plt.tight_layout()

        # Grafikni BytesIO ob'ektiga saqlash
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')

        # JSON javobini qaytarish
        return JsonResponse({'image': image_base64})
# Define file paths and labels


file_paths = [
    settings.BASE_DIR / 'okclass.csv',
    settings.BASE_DIR / 'qisishclass.csv',
    settings.BASE_DIR / 'yoyishclass.csv',
]
labels = ['ok', 'qo\'lni qisish', 'yoyish']

def calculate_metrics(signal_values, selected_features):
    try:
        N = len(signal_values)
        
        if N == 0:
            return {'error': 'Signal values are empty'}

        metrics = {}
        def modify_MAV1(value):
            return abs(value) if abs(value) >= 0.5 else 0.5
        def modify_MAV2(value, index):
            if abs(value) >= 0.25 and abs(value) < 0.75:
                return 0.25 * abs(value)
            else:
                return 4 * index / N
        modified_values2 = np.array([modify_MAV2(x, i) for i, x in enumerate(signal_values)])

        modified_values1 = np.array([modify_MAV1(x) for x in signal_values])
        # Calculate only the requested metrics
        if 'MAV' in selected_features:
            metrics['MAV'] = np.sum(np.abs(signal_values)) / N
        if 'G' in selected_features:
            metrics['G'] = np.sum(signal_values)
        if 'MAV1' in selected_features:
            metrics['MAV1'] = np.sum(modified_values1) / N
        if 'MAV2' in selected_features:
            metrics['MAV2'] = np.sum(modified_values2) / N
        if 'SSI' in selected_features:
            metrics['SSI'] = np.sum(signal_values**2)
        if 'VAR' in selected_features:
            metrics['VAR'] = np.var(signal_values)
        if 'TM3' in selected_features:
            VAR = np.var(signal_values)
            metrics['TM3'] = np.mean(np.abs(signal_values))
        if 'TM5' in selected_features:
            VAR = np.var(signal_values)
            metrics['TM5'] = np.mean(signal_values**2)
        if 'RMS' in selected_features:
            metrics['RMS'] = np.sqrt(np.mean(signal_values**2))
        if 'LOG' in selected_features:
            metrics['LOG'] = np.sum(np.log1p(np.abs(signal_values))) / N  
        if 'WL' in selected_features:
            metrics['WL'] = np.sum(np.abs(np.diff(signal_values)))
        if 'ZC' in selected_features:
            h = np.mean(signal_values)
            metrics['ZC'] = (1 / (2 * N)) * np.sum(np.sign(signal_values - h) - np.sign(np.roll(signal_values, 1) - h))
        if 'AAC' in selected_features:
            metrics['AAC'] = np.sum(np.abs(np.diff(signal_values))) / (N - 1)
        if 'DASDV' in selected_features:
            metrics['DASDV'] = np.sqrt(np.sum((np.diff(signal_values))**2) / (N - 1))
        if 'FFT' in selected_features:
            fft_values = np.fft.fft(signal_values)
            FFT_magnitude = np.abs(fft_values)  # Magnitude of the FFT

            # Variant 1: Maksimal magnitudani saqlash
            FFT_max = np.max(FFT_magnitude)

            # Variant 2: O'rtacha magnitudani saqlash
            FFT_mean = np.mean(FFT_magnitude)

            # Variant 3: Energiyani saqlash
            FFT_energy = np.sum(FFT_magnitude**2)
            metrics['FFT'] = FFT_mean
        if 'PSR' in selected_features:
            fft_values = np.fft.fft(signal_values)
            power_spectrum = np.abs(fft_values) ** 2

            # Signal kuchining umumiy yig'indisi
            total_power = np.sum(power_spectrum)

            # Signalning maksimal chastotasi (dominant frequency) ni topish
            dominant_frequency_power = np.max(power_spectrum)

            # PSR ni hisoblash
            PSR = dominant_frequency_power / total_power if total_power > 0 else 0
            metrics['PSR'] = PSR
        if 'MNF' in selected_features:
            fft_values = np.fft.fft(signal_values)
            FFT_magnitude = np.abs(fft_values)

            # Chastotalar diapazoni bo'yicha o'rtacha hisoblash
            frequencies = np.fft.fftfreq(len(signal_values))
            MNF = np.sum(frequencies * FFT_magnitude) / np.sum(FFT_magnitude)
            metrics['MNF'] =MNF
        
        if 'WAMP' in selected_features:
            metrics['WAMP'] = np.mean(np.abs(signal_values)) 
        if 'IEMG' in selected_features:
            metrics['IEMG'] = np.sum(np.abs(signal_values))
        if 'logDetect' in selected_features:
            metrics['logDetect'] = np.sum(np.log1p(np.abs(signal_values)))
        
        return metrics
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {}

class ClassificationAPIView(APIView):
    parser_classes = [JSONParser]

    def post(self, request):
        
        selected_features = request.data.get('features', [])
        
        if not selected_features:
            return Response({"error": "Features are required."}, status=status.HTTP_400_BAD_REQUEST)

        results = []
        try:
            
            for file_path, label in zip(file_paths, labels):
                
                # if not os.path.exists(file_path):
                #     return Response({"error": f"'{file_path}' fayli topilmadi."}, status=status.HTTP_404_NOT_FOUND)

                data = pd.read_csv(f'{file_path}', header=None)
                
                for column in data.columns:
                    signal_values = data[column].values

                    # Convert signal values to numeric format
                    try:
                        signal_values = signal_values.astype(float)
                    except ValueError:
                        continue

                    metrics = calculate_metrics(signal_values, selected_features)
                    metrics['Label'] = label  # Add label to each file's metrics
                    results.append(metrics)

            # Generate CSV response
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="results.csv"'

            writer = csv.DictWriter(response, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

            return response
        
        except Exception as e:
            print(f"Error processing request: {e}")
            return Response({"error": "An error occurred while processing the request."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Ushbu funksiya modelni o'qitadi va baholaydi.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, classification_rep

@csrf_exempt
def upload_csv(request):
    if request.method == 'POST' and request.FILES.get('file'):
        csv_file = request.FILES['file']

        # Faqat CSV fayl qabul qilishni tekshirish
        if not csv_file.name.endswith('.csv'):
            return JsonResponse({'error': 'Yuklangan fayl CSV formatida emas.'}, status=400)

        # Faylni saqlash
        file_path = default_storage.save(f"uploads/{csv_file.name}", csv_file)
        file_path = os.path.join('media', file_path)  # Media papkasiga yo'lni olish

        try:
            # CSV faylni yuklash
            df = pd.read_csv(file_path)

            # Ma'lumotlarni ajratish: X (xususiyatlar) va y (yorliqlar)
            X = df.drop(columns=['Label'])
            y = df['Label']

            # Ma'lumotlarni o'qitish va test uchun bo'lish
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Modellar ro'yxati
            models = {
                'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
                'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
                'SVC': SVC(kernel='linear', random_state=42),
                'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5)
            }

            # Natijalarni saqlash uchun ro'yxat
            results = []

            # Har bir modelni o'qitish va baholash
            for model_name, model in models.items():
                accuracy, classification_rep = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
                results.append({
                    'model': model_name,
                    'accuracy': accuracy,
                    'classification_report': classification_rep
                })

            # Eng yaxshi modelni aniqlash
            best_model = max(results, key=lambda x: x['accuracy'])

            # Natijalarni qaytarish
            return JsonResponse({
                'message': 'Fayl muvaffaqiyatli yuklandi va qayta ishlangan.',
                'results': results,  # Barcha modellar uchun natijalar
                'best_model': best_model  # Eng yaxshi model
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Noto‘g‘ri so‘rov.'}, status=400)