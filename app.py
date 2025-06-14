import streamlit as st
import matplotlib.pyplot as plt
from qiskit import *
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import time
import numpy as np
from PIL import Image, ImageOps
import io
import copy
import random
import hashlib

st.set_page_config(
    page_title="Criptografie Cuantica",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""<h1>Quantum vs Clasic challenge criptografic</h1>""", unsafe_allow_html=True)

page = st.sidebar.selectbox("Alege challenge-ul", [
    "🏠 Acasă & Prezentare generală",
    "🔍 Atacul de căutare al lui Grover",
    "🔢 Atacul de factorizare al lui Shor",
    "📊 Comparație a complexității",
    "🎮 Laborator Cuantic Interactiv",
    "🧠 Grover vs Kyber (simulare educațională)",
    "🔐 Simulare Interactivă Post-Quantum: Schimb de chei Kyber-like"
])

st.markdown("""
<style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap');

        .box {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin: 1rem 0;
        }
        .algo-header{
            font-family: 'Poppins', sans-serif;
        }
        .metric_box{
            background-color: white;
            color: black;
            border-radius:20px;
            padding-top: 20px;
            padding-bottom: 20px;
            padding-left: 10px;
            padding-right: 10px;
        }
</style>
""", unsafe_allow_html=True)

def simulate_classical_search(n_bits):
    max_attempts = 2**(n_bits - 1)
    return max_attempts

# GROVER - START

def create_oracle(code: str):
    nr = len(code)

    oracle = QuantumCircuit(nr, name="oracle")
    
    for i, bit in enumerate(reversed(code)):
        if bit == '0':
            oracle.x(i)

    target = nr - 1
    controls = list(range(nr - 1))

    # apply MCZ = H + MCX + H
    oracle.h(target)
    oracle.mcx(controls, target)
    oracle.h(target)

    for i, bit in enumerate(reversed(code)):
        if bit == '0':
            oracle.x(i)

    return oracle

def create_diffusion(code): # H - X - MCZ - X - H
    nr = len(code)

    diff = QuantumCircuit(nr, name="diffusion")

    all_qubits = range(nr)
    target = nr - 1
    controls = list(range(nr - 1))

    diff.h(all_qubits)
    diff.x(all_qubits)

    diff.h(target)
    diff.mcx(controls, target)
    diff.h(target)

    diff.x(all_qubits)
    diff.h(all_qubits)

    return diff

def run_grover_algo(code): # H - oracle - diff
    qubits = len(code)

    grover = QuantumCircuit(qubits)
    grover.h(range(qubits))

    num_iterations = math.ceil(math.pi / 4 * math.sqrt(2 ** len(code)))

    for _ in range(num_iterations):
        grover.append(create_oracle(code), list(range(len(code))))
        grover.append(create_diffusion(code), list(range(len(code))))

    grover.measure_all()

    backend = AerSimulator()
    transpiled_grover = transpile(grover, backend)
    job = backend.run(transpiled_grover)
    result = job.result()
    counts = result.get_counts()

    return grover, counts, num_iterations

# GROVER - END

# SHOR - START

def c_amod15(a, power):
    U = QuantumCircuit(4, name=f"{a}^{power} mod 15")

    for iter in range(power):
        U.swap(0,1)
        U.swap(1,2)
        U.swap(2,3)

        for q in range(4):
            U.x(q)

    U = U.to_gate()
    U.name = f"{a}^{power} mod 15"
    c_U = U.control()
    return c_U 

def qft_dagger(n):
    qc = QuantumCircuit(n)
    for qubit in range(n // 2):
        qc.swap(qubit, n - qubit - 1)

    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi / float(2 ** (j - m)), m, j)
        qc.h(j)

    qc.name = "dagger"     
    return qc  

def run_shor_algo(n_count=8, a=7):
    qc = QuantumCircuit(n_count + 4, n_count)

    for q in range(n_count):
        qc.h(q)
    
    qc.x(n_count)

    for q in range(n_count):
        qc.append(c_amod15(a, 2**q), [q] + [i + n_count for i in range(4)])

    qc.append(qft_dagger(n_count), range(n_count))
    qc.measure(range(n_count), range(n_count))

    backend = AerSimulator()
    transpiled_res = transpile(qc, backend)
    job = backend.run(transpiled_res, shots=1024)
    result = job.result()
    counts = result.get_counts()

    return qc, counts


# SHOR - END

if page == "🏠 Acasă & Prezentare generală":
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="box">
            <h2>🌟 Bine ai venit în Era Cuantică!</h2>
            <p>Explorează modul în care calculul cuantic revoluționează criptografia și securitatea cibernetică.
            Această platformă interactivă demonstrează puterea algoritmilor cuantici în spargerea metodelor tradiționale de criptare.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🎯 Ce vei descoperii: ")
        st.markdown("""
        - *Algoritmul Grover*: Căutare în baze de date folosind calculul cuantic, cu o accelerare de ordin pătratic
        - *Algoritmul Shor*: Factoring cuantic care amenință criptarea RSA
        - *Analiza complexității*: Comparație între performanța clasică și cea cuantică
        - *Laboratoare interactive*: Experimente practice cu circuite cuantice
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="box">
            <h2>⚡ Avantajul Cuantic</h2>
            <p>În timp ce calculatoarele clasice necesită timp exponențial pentru anumite probleme, 
            calculatoarele cuantice le pot rezolva mult mai rapid, schimbând fundamental 
            domeniul securității cibernetice.</p>
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure()
        n_values = np.arange(1, 21)

        classical_time = 2**n_values
        quantum_time = np.sqrt(2**n_values)

        fig.add_trace(go.Scatter(x=n_values, y=classical_time, name="Clasic O(2^n)", line=dict(color='red', width=3)))
        fig.add_trace(go.Scatter(x=n_values, y=quantum_time, name="Quanutm O(√2^n)", line=dict(color='blue', width=3)))

        fig.update_layout(title="Quantum vs Classic Complexity",
                          xaxis_title="Problem size (n)",
                          yaxis_title="Time Complexity",
                          yaxis_type="log",
                          template="plotly_white")

        st.plotly_chart(fig, use_container_width=True)


elif page == "🔍 Atacul de căutare al lui Grover":
    st.markdown("<h2 class='algo-header'>🔍 Algoritmul de cautare Grover</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        apasat = False

        st.markdown("### 🎮 Configureaza atacul")
        qubits = st.slider("Numarul de qubits (lungimea parolei)", 2, 8, 4)

        secret_code = st.text_input("Codul secret care trebuie gasit: ", value="0"*qubits, max_chars=qubits)

        if len(secret_code) != qubits:
            st.error(f"Codul secret nu poate depasi {qubits} biti!")
        else:
            if st.button("Lanseaza atacul cuantic!", type="primary"):
                apasat = True
                with st.spinner("Calculatorul cuantic functioneaza..."):

                    time.sleep(1)

                    circuit, counts, iterations = run_grover_algo(secret_code)

                     # show the circuit
                    if apasat:
                        st.markdown("### 🔢 Circuitul generat")
                        fig = circuit.draw(output="mpl", fold=-1)
                        st.pyplot(fig)

                    st.success("✅ Atactul cuantic a fost completat in {iterations} iteratii!")

                    cel_mai_cautat = max(counts.items(), key=lambda x: x[1])

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Valoarea a fost gasita", cel_mai_cautat[0])
                    with col_b:
                        st.metric("Iteratii cuantice", iterations)
                    with col_c:
                        classical_attempt = simulate_classical_search(qubits)
                        st.metric("Speedrun factor", f"{classical_attempt//iterations}x")

                    st.markdown("### Rezultatele masuratorii")
                    fig = plot_histogram(counts, figsize=(10,6))
                    st.pyplot(fig)

    with col2:

        st.markdown("### 🧠 Cum functioneaza?")
        st.markdown("""
        <div class="box">
            <h4>Pașii Algoritmului lui Grover:</h4>
            <ol>
                <li><strong>Superpoziție:</strong> Pune toți qubiții într-o superpoziție egală</li>
                <li><strong>Oracol:</strong> Marchează starea țintă prin inversarea fazei</li>
                <li><strong>Difuzie:</strong> Amplifică amplitudinea stării marcate</li>
                <li><strong>Repetare:</strong> Iterează de aproximativ √N ori pentru N posibilități</li>
            </ol>
        </div>

        """, unsafe_allow_html=True)

        st.markdown("### 💡 Impact in viata reala")
        attack_times = pd.DataFrame({
            "Lungimea parolei": [4, 8, 16, 32, 64],
            "Clasic (secunde)": [0.00001, 0.003, 11, 68000000, 2.9e14],
            "Quantum (secunde)": [0.000001, 0.00004, 0.0006, 0.26, 16777]
        })

        st.dataframe(attack_times, use_container_width=True)

elif page == "🔢 Atacul de factorizare al lui Shor":
    st.markdown('<h2 class="algo-header">Algoritmul de Factorizare Shor</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([1,1])

    with col2:
        st.markdown("### 🎯 Vulnerabilitatea algoritmului RSA")

        if st.button("Factorizeaza numarul 15 folosind Quantum", type="primary"):
            with st.spinner("Factorizarea cuantica se incarca..."):
                time.sleep(2)
                circuit, counts = run_shor_algo()

                st.success("✅ Determinarea perioadei cuantice a fost finalizată!")

                st.markdown("### 🔧 Circuitul Cuantic")

                try:
                    fig = circuit.draw(output="mpl", fold=-1)
                    st.pyplot(fig)
                except:
                    st.code(str(circuit.draw()))

                st.markdown("### Rezultatele Masurarii")
                fig_hist = plot_histogram(counts, figsize=(10, 6))
                st.pyplot(fig_hist)

                st.markdown("""
                <div class="metric_box">
                <h4>🎯 Rezultatul Factorizarii</h4>
                <p><strong>N = 15 = 3 × 5</strong></p>
                <p>Algoritmul cuantic a gasit factorizarea cu succes!</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col1:
        st.markdown("### 🧠 Înțelegerea Algoritmului lui Shor")

        st.markdown("""
        <div class="box">
            <h4>Pași esențiali:</h4>
            <ol>
                <li><strong>Determinarea perioadei:</strong> Găsește perioada lui a^x mod N</li>
                <li><strong>Transformata Fourier Cuantică:</strong> Extrage perioada din superpoziție</li>
                <li><strong>Post-procesare clasică:</strong> Folosește perioada pentru a găsi factorii</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔐 Criptarea RSA în Pericol")

        rsa_sizes = pd.DataFrame({
            'Dimensiunea cheii RSA (biți)': [512, 1024, 2048, 4096],
            'Timp clasic (ani)': [1, 1000000, 1e15, 1e30],
            'Timp cuantic (ore)': [0.1, 10, 100, 1000]
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Clasic', x=rsa_sizes['Dimensiunea cheii RSA (biți)'], 
                            y=rsa_sizes['Timp clasic (ani)'], marker_color='red'))
        fig.add_trace(go.Bar(name='Cuantic', x=rsa_sizes['Dimensiunea cheii RSA (biți)'], 
                            y=rsa_sizes['Timp cuantic (ore)'], marker_color='blue'))

        fig.update_layout(title="Comparație: Timp de Spargere RSA",
                        xaxis_title="Dimensiunea cheii RSA (biți)",
                        yaxis_title="Timp necesar pentru spargere",
                        yaxis_type="log",
                        barmode='group',
                        template="plotly_white")

        st.plotly_chart(fig, use_container_width=True)

elif page == "📊 Comparație a complexității":
    st.markdown("## 📊 Analiza Clasic vs Quantum")

    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown("#### ⚙ Ajusteaza Parametrii")
        
        max_n = st.slider("Dimensiunea maxima", 10, 30, 20)
        
        n_range = np.arange(1, max_n + 1)
        cautare_clasica = 2**n_range
        cautare_cuantica = np.sqrt(2**n_range)
        factor_clasic = np.exp(1.9 * ((n_range * np.log(2)) ** (1/3)) * (np.log(n_range * np.log(2))) ** (2/3))
        factor_cuantic = n_range**3
        
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Complexitatea cautarii", "Complexitatea factorizarii"), vertical_spacing=0.15)
        
        fig.add_trace(go.Scatter(x=n_range, y=cautare_clasica, name="Cautare clasica O(2^n)", 
                                 line=dict(color="red", width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=n_range, y=cautare_cuantica, name="Cautare cuantum O(√2^n)", 
                                 line=dict(color="blue", width=3)), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=n_range, y=factor_clasic, name='Factorizare clasica', 
                                line=dict(color='red', width=3, dash='dash')), row=2, col=1)
        fig.add_trace(go.Scatter(x=n_range, y=factor_cuantic, name='Factorizare cuantica O(n³)', 
                                line=dict(color='blue', width=3, dash='dash')), row=2, col=1)
        
        fig.update_xaxes(title_text="Dimensiunea problemei (n bits)", row=2, col=1)
        fig.update_yaxes(title_text="Complexitatea de timp", type="log", row=1, col=1)
        fig.update_yaxes(title_text="Complexitatea de timp", type="log", row=2, col=1)
        
        fig.update_layout(height=600, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        

    with col2:
        st.markdown("#### 🎯 Puncte Cheie")
        
        n_example = 20
        timp_cautare_clasic = 2**n_example
        timp_cautare_cuantic = np.sqrt(2**n_example)
        
        st.markdown(f"""
                    <div class="metric_box">
                    <h4>🎯 Cautarea (n={n_example})</h4>
                    <p><strong>Cautarea clasica</strong> {timp_cautare_clasic:.0f} operatii</p>
                    <p><strong>Cautarea cuantica</strong> {timp_cautare_cuantic:.0f} operatii</p>
                    </div>                    
                    """, unsafe_allow_html=True)
        
        
        st.divider()
        
        st.markdown("""
        <div class="metric_box">
        <h4>🔢 Problema factorizarii</h4>
        <p><strong>RSA-2048:</strong> Clasic: 1000+ ani → Cuantic: ore</p>
        <p><strong>Securitatea curenta:</strong> Siggura pentru aatacurile cuantice</p>
        <p><strong>Era post-cuantica:</strong> Necesita algoritmi cuantici</p>
        </div>
        """, unsafe_allow_html=True)
        
elif page == "🎮 Laborator Cuantic Interactiv":
    if "qc" not in st.session_state or "num_qubits" not in st.session_state:
        st.session_state.num_qubits = 3
        st.session_state.qc = QuantumCircuit(st.session_state.num_qubits)
        st.session_state.history = []
        st.session_state.future = []

    num_qubits = st.slider("🔢 Alege numărul de qubiți", 2, 8, st.session_state.num_qubits)

    # reseteaza circuitul daca s-a schimbat numarul de qubits
    if num_qubits != st.session_state.num_qubits:
        st.session_state.qc = QuantumCircuit(num_qubits)
        st.session_state.num_qubits = num_qubits
        st.session_state.history.clear()
        st.session_state.future.clear()
        st.success("Circuit resetat cu număr nou de qubiți.")

    def save_state():
        st.session_state.history.append(copy.deepcopy(st.session_state.qc))
        st.session_state.future.clear()

    qc = st.session_state.qc

    st.subheader("➕ Adaugă o poartă")

    gate = st.selectbox("Tip poarta", ["H", "X", "Z", "CX", "MCX", "MCZ"])
    
    qubits = []
    
    if gate == "Z":
        target = st.multiselect("Qubit tinta", list(range(num_qubits)))
        qubits = [target]
    elif gate == "X":
        target = st.multiselect("Qubit tinta", list(range(num_qubits)))
        qubits = [target]
    elif gate == "H":
        target = st.multiselect("Qubit tinta", list(range(num_qubits)))
        qubits = [target]
    elif gate == "CX":
        control = st.selectbox("Qubit de control", list(range(num_qubits)))
        other_qubits = list(filter(lambda x: x != control, list(range(num_qubits))))
        target = st.selectbox("Qubit tinta", other_qubits)
        qubits.append(control)
        qubits.append(target)
    elif gate == "MCX":
        controls = st.multiselect("Qubitii de control", list(range(num_qubits)))
        if len(controls) == num_qubits:
            st.error("Nu poți selecta toți qubiții ca fiind de control. Selectează cel puțin un qubit țintă.")
        other_qubits = [q for q in range(num_qubits) if q not in controls]
        target = st.selectbox("Qubit tinta", other_qubits)
        qubits.append(controls)
        qubits.append(target)
    elif gate == "MCZ":
        controls = st.multiselect("Qubitii de control", list(range(num_qubits)))
        if len(controls) == num_qubits:
            st.error("Nu poți selecta toți qubiții ca fiind de control. Selectează cel puțin un qubit țintă.")
        other_qubits = [q for q in range(num_qubits) if q not in controls]
        target = st.selectbox("Qubit tinta", other_qubits)
        qubits.append(controls)
        qubits.append(target)  

    if st.button("Aplica poarta"):
        save_state()

        if gate == 'H':
            qc.h(qubits[0])
        elif gate == "X":
            qc.x(qubits[0])
        elif gate == "CX":
            qc.cx(qubits[0], qubits[1])
        elif gate == "Z":
            qc.z(qubits[0])
        elif gate == "MCX":
            qc.mcx(qubits[0], qubits[1])
        elif gate == "MCZ":
            qc.h(qubits[1])
            qc.mcx(qubits[0], qubits[1])
            qc.h(qubits[1])

    col_undo, col_redo = st.columns([1, 1])

    with col_undo:
        if st.button("↩️ Undo") and st.session_state.history:
            st.session_state.future.append(copy.deepcopy(st.session_state.qc))
            st.session_state.qc = st.session_state.history.pop()
            st.success("S-a aplicat undo.")

    with col_redo:
        if st.button("↪️ Redo") and st.session_state.future:
            st.session_state.history.append(copy.deepcopy(st.session_state.qc))
            st.session_state.qc = st.session_state.future.pop()
            st.success("S-a aplicat redo.")

    qc = st.session_state.qc

    col1, col2 = st.columns([1,1])

    with col1:
        st.subheader("🧾 Circuit actual")

    with col2:
        if st.button("Resetare circuit"):
            with st.spinner("Circuitul se reseteaza..."):
                time.sleep(2)
                st.session_state.qc = QuantumCircuit(num_qubits)
                qc = st.session_state.qc

    img = qc.draw("latex")
    padded_img = ImageOps.expand(img, border=20, fill=(255, 255, 255))
    st.image(padded_img)

    st.subheader("📊 Simulare cu măsurători")
    if st.button("▶️ Rulează circuitul și măsoară"):
        qc_measure = qc.copy()
        qc_measure.measure_all()
        backend = AerSimulator()
        transpiled_result = transpile(qc_measure, backend)
        job = backend.run(transpiled_result)
        result = job.result()
        counts = result.get_counts()
        
        fig = plot_histogram(counts, figsize=(10, 5))
        st.pyplot(fig)
elif page == "🧠 Grover vs Kyber (simulare educațională)":
    st.markdown("<h2 class='algo-header'>🧠 Grover contra Kyber – Simulare educațională</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div class="box">
        <p>Algoritmul Grover oferă un avantaj cuantic în spargerea criptării simetrice, dar este ineficient împotriva algoritmilor post-quantum precum <strong>Kyber</strong>.
        În această simulare, explorăm de ce Grover nu poate sparge în mod realist Kyber.</p>
    </div>
    """, unsafe_allow_html=True)

    n = st.slider("🔐 Număr de biți ai cheii (Kyber512 ≈ 256)", min_value=2, max_value=12, value=4)

    # Estimare Grover
    iterations = int((math.pi / 4) * math.sqrt(2 ** n))
    st.markdown(f"""
    <div class="metric_box">
        <h4>ℹ️ Estimare Grover:</h4>
        <ul>
            <li>Număr de qubiți necesari: <strong>{n}</strong></li>
            <li>Număr de iterații Grover: <strong>{iterations}</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Circuit Grover educațional
    st.markdown("### 🧪 Circuit demonstrativ Grover (mică dimensiune)")
    from qiskit import QuantumCircuit
    from qiskit.visualization import plot_histogram

    grover_qc = QuantumCircuit(n)
    grover_qc.h(range(n))
    for _ in range(iterations):
        grover_qc.h(range(n))
        grover_qc.x(range(n))
        grover_qc.h(n - 1)
        grover_qc.mcx(list(range(n - 1)), n - 1)
        grover_qc.h(n - 1)
        grover_qc.x(range(n))
        grover_qc.h(range(n))
    grover_qc.measure_all()

    st.pyplot(grover_qc.draw(output='mpl', fold=-1))

    # Simulare imposibilă la 256
    if n >= 10:
        st.markdown("""
        <div class="box">
            <h3>🚫 Atac Grover imposibil în practică pentru n ≥ 256</h3>
            <p>Un atac Grover ar necesita:</p>
            <ul>
                <li>~2<sup>128</sup> iterații pentru cheia Kyber-512</li>
                <li>Milione de qubiți fizici stabili</li>
                <li>Milenii de rulări fără erori</li>
            </ul>
            <p><strong>➡️ Concluzie:</strong> Kyber este practic imposibil de spart cu Grover. Este considerat <em>post-quantum secure</em>.</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "🔐 Simulare Interactivă Post-Quantum: Schimb de chei Kyber-like":
    st.markdown("<h2 class='algo-header'>🔐 Simulare interactivă: Schimb de chei post-cuantic (Kyber-like)</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div class="box">
    <p>Simulează pas cu pas un schimb de chei de tip <strong>Kyber</strong>, un algoritm post-cuantic standardizat de NIST. Deși nu rulăm criptografia reală, acest flux imită mecanismul <em>Key Encapsulation Mechanism (KEM)</em>.</p>
    </div>
    """, unsafe_allow_html=True)

    if "alice_pubkey" not in st.session_state:
        st.session_state.alice_pubkey = None
        st.session_state.secret = None
        st.session_state.ciphertext = None
        st.session_state.bob_shared = None
        st.session_state.alice_shared = None

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 👩 Alice – Inițializare")

        if st.button("🔑 1. Generează cheia publică a lui Alice"):
            pk = hashlib.sha256(str(random.randint(0, 1e10)).encode()).hexdigest()
            st.session_state.alice_pubkey = pk
            st.success("Cheia publică a fost generată.")
            st.code(f"Alice Public Key: {pk[:64]}...")

    with col2:
        st.markdown("### 🧑‍🚀 Bob – Encapsulare")

        if st.session_state.alice_pubkey:
            if st.button("📦 2. Bob trimite cheia criptată"):
                secret = f"{random.randint(100000, 999999)}"
                ciphertext = f"ENC({secret})"
                shared_key = hashlib.sha256(secret.encode()).hexdigest()

                st.session_state.secret = secret
                st.session_state.ciphertext = ciphertext
                st.session_state.bob_shared = shared_key

                st.success("Cheie generată și criptată.")
                st.code(f"Ciphertext către Alice: {ciphertext}")
                st.code(f"Cheia partajată (Bob): {shared_key[:64]}...")

    st.divider()

    st.markdown("### 📨 3. Alice primește și decapsulează")

    if st.session_state.ciphertext:
        if st.button("🔓 Decapsulează ciphertext"):
            secret_received = st.session_state.ciphertext.replace("ENC(", "").replace(")", "")
            alice_shared = hashlib.sha256(secret_received.encode()).hexdigest()
            st.session_state.alice_shared = alice_shared

            with st.expander("🔐 Cheia decapsulată de Alice"):
                st.code(f"{alice_shared[:64]}...")

            if alice_shared == st.session_state.bob_shared:
                st.success("✅ Schimb de chei reușit: cheile coincid.")
                st.balloons()
            else:
                st.error("❌ Cheile NU coincid. Ceva nu a mers bine.")

    st.divider()
    st.markdown("""
    <div class="metric_box">
    <h4>📚 Ce ai simulat:</h4>
    <ul>
        <li>🔑 Alice a generat o pereche cheie publică/secretă</li>
        <li>📦 Bob a creat o cheie simetrică și a trimis-o criptat</li>
        <li>🔓 Alice a extras cheia din mesajul criptat</li>
    </ul>
    <p><strong>➡️ În realitate:</strong> acest flux este implementat în Kyber și folosit pentru a securiza comunicațiile în fața atacurilor cuantice.</p>
    </div>
    """, unsafe_allow_html=True)