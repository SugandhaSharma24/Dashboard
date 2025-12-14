import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="EMVO AI Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Dark Theme CSS
st.markdown("""
<style>
    :root {
        --primary: #8B5CF6;
        --primary-light: #A78BFA;
        --primary-dark: #7C3AED;
        --secondary: #0EA5E9;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --dark-bg: #0F172A;
        --dark-card: #1E293B;
        --dark-border: #334155;
        --light-text: #F8FAFC;
        --light-text-secondary: #CBD5E1;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--dark-bg) 0%, #1E293B 100%);
        color: var(--light-text);
    }
    
    .stMetric {
        background: linear-gradient(135deg, var(--dark-card) 0%, #1A2332 100%) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        border: 1px solid var(--dark-border) !important;
    }
    
    .stMetric label {
        color: var(--light-text-secondary) !important;
    }
    
    .stMetric div[data-testid="stMetricValue"] {
        color: white !important;
    }
    
    .stMetric div[data-testid="stMetricDelta"] {
        color: var(--light-text-secondary) !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--primary-light) 0%, var(--primary) 100%);
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.3);
        transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--dark-card);
        color: var(--light-text-secondary);
        border: 1px solid var(--dark-border);
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        border-color: var(--primary);
    }
    
    .stTextInput > div > div > input {
        background: var(--dark-card);
        color: var(--light-text);
        border: 1px solid var(--dark-border);
        border-radius: 8px;
    }
    
    .stSelectbox > div > div {
        background: var(--dark-card);
        color: var(--light-text);
        border: 1px solid var(--dark-border);
        border-radius: 8px;
    }
    
    .stSlider > div > div > div {
        background: var(--dark-card);
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, var(--dark-bg) 100%);
        border-right: 1px solid var(--dark-border);
    }
    
    section[data-testid="stSidebar"] .stRadio > div {
        background: transparent;
    }
    
    section[data-testid="stSidebar"] label {
        color: var(--light-text) !important;
        font-weight: 500;
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.25rem 0;
    }
    
    section[data-testid="stSidebar"] label:hover {
        background: rgba(139, 92, 246, 0.1);
    }
    
    .stDataFrame {
        background: var(--dark-card);
        border: 1px solid var(--dark-border);
        border-radius: 8px;
    }
    
    .stAlert {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid var(--dark-border) !important;
        color: var(--light-text) !important;
    }
    
    .stExpander {
        background: var(--dark-card);
        border: 1px solid var(--dark-border);
        border-radius: 8px;
    }
    
    .stExpander > div > div {
        background: var(--dark-card) !important;
    }
    
    .chat-bubble-user {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }
    
    .chat-bubble-agent {
        background: linear-gradient(135deg, #2D3748 0%, #1A2332 100%);
        color: var(--light-text);
        border-radius: 18px 18px 18px 4px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        max-width: 80%;
        border: 1px solid var(--dark-border);
    }
    
    .chat-container {
        background: linear-gradient(135deg, #1A2332 0%, var(--dark-card) 100%);
        border-radius: 12px;
        padding: 1.5rem;
        height: 400px;
        overflow-y: auto;
        border: 1px solid var(--dark-border);
        margin-bottom: 1rem;
    }
    
    .threat-card {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .warning-card {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .safe-card {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_agent' not in st.session_state:
    st.session_state.selected_agent = None

# Sample data for agents
def get_agents_data():
    return {
        "Customer Support Bot": {
            "status": "active",
            "type": "Customer Service",
            "conversations": 1245,
            "success_rate": 96.2,
            "avg_response_time": 1.8,
            "template": "Agentic Customer Support",
            "chat_history": [
                {"user": "Hello, I need help with my order", "agent": "I'd be happy to help! Could you share your order number?", "time": "10:30 AM"},
                {"user": "ORD-789012", "agent": "Found it! Your order shipped yesterday and will arrive tomorrow.", "time": "10:31 AM"}
            ],
            "analytics": {
                "daily_interactions": [45, 52, 48, 67, 58, 61, 55],
                "satisfaction_scores": [4.8, 4.7, 4.9, 4.6, 4.8, 4.7, 4.9]
            },
            "uptime": "99.8%",
            "last_active": "2 mins ago"
        },
        "Data Analyst Agent": {
            "status": "active",
            "type": "Data Analysis",
            "conversations": 892,
            "success_rate": 94.5,
            "avg_response_time": 3.2,
            "template": "Automated Data Analysis",
            "chat_history": [
                {"user": "Show me last week's sales", "agent": "Generating sales report for last week...", "time": "11:15 AM"},
                {"user": "Compare with previous month", "agent": "Last week: $125K, Previous month same week: $98K (+27.5%)", "time": "11:16 AM"}
            ],
            "analytics": {
                "daily_interactions": [28, 31, 25, 34, 29, 32, 30],
                "satisfaction_scores": [4.5, 4.6, 4.4, 4.7, 4.5, 4.6, 4.8]
            },
            "uptime": "99.5%",
            "last_active": "5 mins ago"
        },
        "Security Monitor": {
            "status": "warning",
            "type": "Security",
            "conversations": 567,
            "success_rate": 98.7,
            "avg_response_time": 0.9,
            "template": "Security Monitoring Agent",
            "chat_history": [
                {"user": "Any threats detected?", "agent": "All systems normal. No threats in last 24 hours.", "time": "09:45 AM"},
                {"user": "Run full scan", "agent": "Initiating full security scan... Estimated completion in 15 minutes.", "time": "09:46 AM"}
            ],
            "analytics": {
                "daily_interactions": [15, 18, 12, 20, 16, 19, 17],
                "satisfaction_scores": [4.9, 4.8, 4.9, 4.7, 4.9, 4.8, 4.9]
            },
            "uptime": "100%",
            "last_active": "Just now"
        },
        "HR Assistant": {
            "status": "active",
            "type": "Human Resources",
            "conversations": 743,
            "success_rate": 91.8,
            "avg_response_time": 2.4,
            "template": "HR Support Agent",
            "chat_history": [
                {"user": "How do I request vacation?", "agent": "You can request vacation through the HR portal. Need the link?", "time": "14:20 PM"},
                {"user": "Yes please", "agent": "Here's the link: https://hrportal.company.com/vacation", "time": "14:21 PM"}
            ],
            "analytics": {
                "daily_interactions": [32, 35, 28, 41, 36, 38, 34],
                "satisfaction_scores": [4.3, 4.4, 4.2, 4.5, 4.3, 4.4, 4.6]
            },
            "uptime": "98.9%",
            "last_active": "30 mins ago"
        }
    }

# Title
st.markdown("## 🤖 EMVO Agentic AI Platform")
st.markdown("##### Production-Ready AI Agent Management & Monitoring System")

# Sidebar navigation
st.sidebar.title("🔍 Navigation")
section = st.sidebar.radio(
    "Select Dashboard:",
    ["🏠 Dashboard", "🤖 Agentverse", "🧠 Brains", "⚖️ Judge"]
)

# Dashboard Section
if section == "🏠 Dashboard":
    st.markdown("## 📊 Agent Observatory Dashboard")
    st.markdown("### Real-time monitoring across all company agents")
    
    agents = get_agents_data()
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_agents = len(agents)
    active_agents = sum(1 for a in agents.values() if a['status'] == 'active')
    total_conversations = sum(a['conversations'] for a in agents.values())
    avg_success_rate = np.mean([a['success_rate'] for a in agents.values()])
    
    with col1:
        st.metric("Total Agents", total_agents, f"{active_agents} active")
    with col2:
        st.metric("Total Conversations", f"{total_conversations:,}", "+124 this week")
    with col3:
        st.metric("Avg Success Rate", f"{avg_success_rate:.1f}%", "+1.2%")
    with col4:
        avg_response = np.mean([a['avg_response_time'] for a in agents.values()])
        st.metric("Avg Response Time", f"{avg_response:.1f}s", "-0.3s")
    
    st.divider()
    
    # Agent Status Overview
    st.markdown("### Agent Status Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create agent status chart
        agent_names = list(agents.keys())
        status_colors = {'active': '#10B981', 'warning': '#F59E0B'}
        
        fig = go.Figure(data=[
            go.Bar(name='Conversations', 
                   x=agent_names, 
                   y=[agents[name]['conversations'] for name in agent_names],
                   marker_color=[status_colors[agents[name]['status']] for name in agent_names]),
            go.Scatter(name='Success Rate', 
                      x=agent_names, 
                      y=[agents[name]['success_rate'] for name in agent_names],
                      yaxis='y2',
                      mode='lines+markers',
                      line=dict(color='#8B5CF6', width=3))
        ])
        
        fig.update_layout(
            title='Agent Performance Metrics',
            template='plotly_dark',
            yaxis=dict(title='Total Conversations'),
            yaxis2=dict(title='Success Rate (%)', overlaying='y', side='right'),
            height=400,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Agent Status")
        for agent_name, agent_data in agents.items():
            status_icon = "🟢" if agent_data['status'] == 'active' else "🟡"
            with st.container():
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.markdown(f"**{agent_name}**")
                    st.markdown(f"*{agent_data['type']}*")
                with col_b:
                    st.markdown(f"{status_icon}")
                st.markdown(f"**Success:** {agent_data['success_rate']}%")
                st.markdown(f"**Uptime:** {agent_data['uptime']}")
                st.divider()
    
    # System Health Metrics
    st.markdown("### System Health Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Security Health
        fig_security = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 87,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Security Score", 'font': {'color': 'white'}},
            gauge = {
                'axis': {'range': [None, 100], 'tickcolor': "white"},
                'bar': {'color': "#8B5CF6"},
                'steps': [
                    {'range': [0, 60], 'color': "#EF4444"},
                    {'range': [60, 80], 'color': "#F59E0B"},
                    {'range': [80, 100], 'color': "#10B981"}
                ],
            }
        ))
        fig_security.update_layout(height=250, template='plotly_dark')
        st.plotly_chart(fig_security, use_container_width=True)
    
    with col2:
        # Performance Health
        fig_perf = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 94,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Performance Score", 'font': {'color': 'white'}},
            gauge = {
                'axis': {'range': [None, 100], 'tickcolor': "white"},
                'bar': {'color': "#0EA5E9"},
                'steps': [
                    {'range': [0, 70], 'color': "#EF4444"},
                    {'range': [70, 85], 'color': "#F59E0B"},
                    {'range': [85, 100], 'color': "#10B981"}
                ],
            }
        ))
        fig_perf.update_layout(height=250, template='plotly_dark')
        st.plotly_chart(fig_perf, use_container_width=True)
    
    with col3:
        # Reliability Health
        fig_reliability = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 98,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Reliability Score", 'font': {'color': 'white'}},
            gauge = {
                'axis': {'range': [None, 100], 'tickcolor': "white"},
                'bar': {'color': "#10B981"},
                'steps': [
                    {'range': [0, 90], 'color': "#EF4444"},
                    {'range': [90, 95], 'color': "#F59E0B"},
                    {'range': [95, 100], 'color': "#10B981"}
                ],
            }
        ))
        fig_reliability.update_layout(height=250, template='plotly_dark')
        st.plotly_chart(fig_reliability, use_container_width=True)

# Agentverse Section
elif section == "🤖 Agentverse":
    st.markdown("## 🤖 EMVO Agentverse")
    st.markdown("### Interactive interface for all company agents")
    
    agents = get_agents_data()
    
    # Agent selection
    st.markdown("### Select an Agent")
    
    cols = st.columns(4)
    for idx, (agent_name, agent_data) in enumerate(agents.items()):
        with cols[idx % 4]:
            status_icon = "🟢" if agent_data['status'] == 'active' else "🟡"
            if st.button(f"{status_icon} **{agent_name}**\n\n*{agent_data['type']}*\n\n**Success:** {agent_data['success_rate']}%", 
                        key=f"select_{agent_name}",
                        use_container_width=True):
                st.session_state.selected_agent = agent_name
                st.rerun()
    
    st.divider()
    
    # Selected agent interface
    if st.session_state.selected_agent:
        agent_name = st.session_state.selected_agent
        agent_data = agents[agent_name]
        
        st.markdown(f"### 🎯 {agent_name} - Interactive Interface")
        
        # Tabs for different interfaces
        tab1, tab2, tab3 = st.tabs(["💬 Chat Interface", "🎤 Voice Interface", "📊 Agent Analytics"])
        
        with tab1:
            st.markdown(f"**Template:** {agent_data['template']} | **Status:** {agent_data['status']}")
            
            # Chat container
            for chat in agent_data['chat_history']:
                  with st.chat_message("user"):
                     st.caption(f"You ({chat['time']})")
                     st.write(chat["user"])

                  with st.chat_message("assistant"):
                     st.caption(agent_name)
                     st.write(chat["agent"])

            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # New message input
            col1, col2 = st.columns([4, 1])
            with col1:
                new_message = st.text_input("Type your message:", key="new_message", placeholder="Type your message here...")
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Send", use_container_width=True):
                    if new_message:
                        st.success(f"Message sent to {agent_name}!")
        
        with tab2:
            st.markdown("### 🎤 Voice Interface")
            st.info("Voice interface integrates with Beam AI's agentic patterns for real-time voice support")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Voice Controls")
                if st.button("🎤 Start Voice Session", use_container_width=True):
                    st.success("Voice session started with agent!")
                
                if st.button("⏸️ Pause Voice", use_container_width=True):
                    st.info("Voice paused")
                
                if st.button("⏹️ End Session", use_container_width=True):
                    st.info("Voice session ended")
            
            with col2:
                st.markdown("#### Voice Settings")
                voice_type = st.selectbox("Voice Type:", ["Natural", "Professional", "Friendly", "Neutral"])
                st.slider("Speech Rate:", 0.5, 2.0, 1.0)
                st.checkbox("Enable real-time transcription", value=True)
        
        with tab3:
            st.markdown(f"### Analytics for {agent_name}")
            
            # Performance metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Success Rate", f"{agent_data['success_rate']}%", f"{agent_data['success_rate'] - 94:.1f}%")
            with col2:
                st.metric("Avg Response Time", f"{agent_data['avg_response_time']}s", f"-{agent_data['avg_response_time'] - 2:.1f}s")
            with col3:
                st.metric("Total Conversations", agent_data['conversations'], "+124")
            
            # Analytics charts
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                y=agent_data['analytics']['daily_interactions'],
                mode='lines+markers',
                name='Daily Interactions',
                line=dict(color='#8B5CF6', width=3)
            ))
            fig1.update_layout(
                title='Weekly Interaction Trends',
                height=300,
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            fig2 = go.Figure(data=[
                go.Bar(
                    x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                    y=agent_data['analytics']['satisfaction_scores'],
                    marker_color='#10B981'
                )
            ])
            fig2.update_layout(
                title='Daily Satisfaction Scores (1-5 scale)',
                height=300,
                yaxis=dict(range=[0, 5]),
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    else:
        st.info("👆 Select an agent from above to interact with it")

# Brains Section
elif section == "🧠 Brains":
    st.markdown("## 🧠 EMVO Brains")
    st.markdown("### Auto LLM Platform for Fine-tuning & Optimization")
    
    # Search section
    st.markdown("#### 🔍 Search for Auto LLM Platforms")
    
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        platform_search = st.text_input("Search platforms or models:", placeholder="e.g., GPT-4, Llama 3, Claude, fine-tuning...")
    
    # Available platforms table
    platforms = pd.DataFrame({
        'Platform': ['OpenAI Fine-Tuning', 'Anthropic Claude', 'Llama 3 70B', 'Gemini Pro', 'Mistral Large'],
        'Type': ['Commercial', 'Commercial', 'Open Source', 'Commercial', 'Commercial'],
        'Context Window': ['128K', '200K', '8K', '1M', '32K'],
        'Best For': ['General purpose', 'Safety & documents', 'Customization', 'Multimodal', 'Efficiency'],
        'Cost/1K Tokens': ['$0.03', '$0.06', 'Free', '$0.001', '$0.008']
    })
    
    if platform_search:
        filtered_platforms = platforms[platforms['Platform'].str.contains(platform_search, case=False)]
    else:
        filtered_platforms = platforms
    
    st.dataframe(filtered_platforms, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Fine-tuning configuration
    st.markdown("#### ⚙️ Fine-tuning Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Hyperparameters")
        selected_model = st.selectbox("Base Model:", ["GPT-4 Turbo", "Llama 3 70B", "Claude 3 Opus", "Gemini Pro", "Custom"])
        
        col1a, col1b = st.columns(2)
        with col1a:
            learning_rate = st.slider("Learning Rate", 1e-5, 1e-3, 5e-4, format="%.5f")
            batch_size = st.select_slider("Batch Size", [8, 16, 32, 64, 128], 32)
        with col1b:
            epochs = st.slider("Epochs", 1, 20, 3)
            warmup_steps = st.number_input("Warmup Steps", 0, 1000, 100)
        
        optimizer = st.selectbox("Optimizer:", ["AdamW", "SGD", "Adafactor", "Adam"])
        scheduler = st.selectbox("Learning Rate Scheduler:", ["Linear", "Cosine", "Constant", "WarmupLinear"])
    
    with col2:
        st.markdown("##### Training Data")
        dataset_type = st.radio("Dataset Type:", ["Structured", "Unstructured", "Mixed"])
        
        if dataset_type == "Structured":
            num_examples = st.number_input("Number of examples:", 100, 100000, 1000)
            st.text_area("Data schema:", value='{"prompt": "", "completion": "", "metadata": {}}', height=100)
        elif dataset_type == "Unstructured":
            uploaded_files = st.file_uploader("Upload training files:", 
                                            accept_multiple_files=True,
                                            type=['txt', 'json', 'csv'])
            if uploaded_files:
                st.write(f"📁 {len(uploaded_files)} files uploaded")
        else:
            st.number_input("Structured examples:", 100, 50000, 1000)
            uploaded_files = st.file_uploader("Upload unstructured files:", 
                                            accept_multiple_files=True,
                                            type=['txt', 'pdf'])
        
        validation_split = st.slider("Validation Split %", 5, 40, 20)
    
    st.divider()
    
    # Training performance monitor
    st.markdown("#### 📈 Training Performance Monitor")
    
    # Simulated training metrics
    epochs_list = list(range(1, epochs + 1))
    train_loss = [2.5 - (2.3 * (e-1)/(epochs-1)) + np.random.normal(0, 0.1) for e in epochs_list]
    val_loss = [2.4 - (2.2 * (e-1)/(epochs-1)) + np.random.normal(0, 0.15) for e in epochs_list]
    accuracy = [0.65 + (0.25 * (e-1)/(epochs-1)) + np.random.normal(0, 0.05) for e in epochs_list]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs_list, y=train_loss, mode='lines+markers', name='Training Loss', line=dict(color='#8B5CF6')))
    fig.add_trace(go.Scatter(x=epochs_list, y=val_loss, mode='lines+markers', name='Validation Loss', line=dict(color='#0EA5E9')))
    fig.add_trace(go.Scatter(x=epochs_list, y=accuracy, mode='lines+markers', name='Accuracy', 
                            yaxis='y2', line=dict(color='#10B981')))
    
    fig.update_layout(
        title='Training Progress',
        template='plotly_dark',
        xaxis=dict(title='Epoch'),
        yaxis=dict(title='Loss'),
        yaxis2=dict(title='Accuracy', overlaying='y', side='right'),
        height=400,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Training metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Epoch", f"{epochs_list[-1]}/{epochs}", "+1")
    with col2:
        st.metric("Training Loss", f"{train_loss[-1]:.3f}", "-0.12")
    with col3:
        st.metric("Validation Loss", f"{val_loss[-1]:.3f}", "-0.08")
    with col4:
        st.metric("Accuracy", f"{accuracy[-1]*100:.1f}%", "+2.3%")
    
    if st.button("🚀 Start Fine-tuning", type="primary", use_container_width=True):
        st.success("Fine-tuning job started! Training performance will be monitored in real-time.")

# Judge Section (LLM Security Research)
elif section == "⚖️ Judge":
    st.markdown("## ⚖️ LLM as Judge Research")
    st.markdown("### Security Assessment & Threat Intelligence Platform")
    
    # Security Dashboard Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Threats", "42", "+8 this week")
    with col2:
        st.metric("Critical Vulnerabilities", "7", "High Priority")
    with col3:
        st.metric("Security Score", "87/100", "-3 from last week")
    with col4:
        st.metric("Protected Models", "12/15", "3 models at risk")
    
    st.divider()
    
    # Threat Categories Analysis
    st.markdown("#### 🚨 Threat Category Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Threat distribution chart
        threat_categories = {
            'Jailbreak Attacks': 42,
            'Prompt Injection': 38,
            'Data Leakage': 25,
            'Model Evasion': 19,
            'Adversarial Examples': 15,
            'Bias Exploitation': 12
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(threat_categories.keys()),
                y=list(threat_categories.values()),
                marker_color=['#EF4444', '#F59E0B', '#8B5CF6', '#0EA5E9', '#10B981', '#6B7280']
            )
        ])
        
        fig.update_layout(
            title='Threat Distribution by Category',
            template='plotly_dark',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Threat Severity")
        
        # Threat severity indicators
        severity_data = [
            {"level": "Critical", "count": 7, "color": "#EF4444"},
            {"level": "High", "count": 12, "color": "#F59E0B"},
            {"level": "Medium", "count": 15, "color": "#0EA5E9"},
            {"level": "Low", "count": 8, "color": "#10B981"}
        ]
        
        for severity in severity_data:
            st.markdown(f"""
            <div style='margin-bottom: 1rem; padding: 1rem; background: rgba(30, 41, 59, 0.5); border-radius: 8px; border: 1px solid {severity['color']};'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span style='color: white; font-weight: 600;'>{severity['level']}</span>
                    <span style='color: {severity['color']}; font-weight: 700; font-size: 1.2rem;'>{severity['count']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # LLM Security Evaluation
    st.markdown("#### 🔍 LLM Security Evaluation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Evaluation Parameters")
        evaluation_type = st.selectbox("Evaluation Type:", 
                                      ["Safety & Alignment", "Adversarial Robustness", 
                                       "Privacy Protection", "Bias Detection", "Comprehensive"])
        
        st.markdown("**Test Categories:**")
        col1a, col1b = st.columns(2)
        with col1a:
            st.checkbox("Jailbreak attempts", True)
            st.checkbox("Prompt injection", True)
            st.checkbox("Data leakage", True)
        with col1b:
            st.checkbox("Adversarial examples", True)
            st.checkbox("Bias exploitation", True)
            st.checkbox("Model evasion", True)
        
        test_intensity = st.slider("Test Intensity:", 1, 10, 7)
        num_test_cases = st.number_input("Test cases:", 10, 10000, 100)
    
    with col2:
        st.markdown("##### Model Comparison")
        models_to_test = st.multiselect("Select models to test:",
                                       ["GPT-4", "Claude 3", "Llama 3", "Gemini Pro", "Mistral Large"],
                                       default=["GPT-4", "Claude 3"])
        
        benchmark_metrics = st.multiselect("Security metrics:",
                                          ["Safety Score", "Robustness", "Privacy", "Fairness", 
                                           "Transparency", "Accountability"],
                                          default=["Safety Score", "Robustness", "Privacy"])
        
        if st.button("🔬 Run Security Assessment", use_container_width=True, type="primary"):
            st.success(f"Starting {evaluation_type} security assessment on {len(models_to_test)} models...")
    
    st.divider()
    
    # Security Insights
    st.markdown("#### 🛡️ Security Insights & Recommendations")
    
    insights = [
        {
            "title": "Jailbreak Prevention",
            "description": "Implement input sanitization and role-based access controls to prevent model jailbreaking attempts.",
            "severity": "Critical",
            "status": "🟡 In Progress"
        },
        {
            "title": "Prompt Injection Defense",
            "description": "Deploy prompt validation layers and context monitoring to detect and block injection attacks.",
            "severity": "High",
            "status": "🟢 Implemented"
        },
        {
            "title": "Data Privacy Protection",
            "description": "Implement differential privacy and data anonymization techniques to prevent training data extraction.",
            "severity": "High",
            "status": "🟡 Planning"
        },
        {
            "title": "Adversarial Robustness",
            "description": "Train models with adversarial examples and deploy input transformation defenses.",
            "severity": "Medium",
            "status": "🟡 Research"
        }
    ]
    
    for insight in insights:
        severity_color = "#EF4444" if insight['severity'] == "Critical" else "#F59E0B" if insight['severity'] == "High" else "#0EA5E9"
        
        st.markdown(f"""
        <div style='margin-bottom: 1rem; padding: 1.5rem; background: rgba(30, 41, 59, 0.5); border-radius: 8px; border: 1px solid {severity_color};'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;'>
                <div style='font-weight: 600; color: white; font-size: 1.1rem;'>{insight['title']}</div>
                <div style='display: flex; gap: 1rem; align-items: center;'>
                    <span style='color: {severity_color}; font-weight: 600;'>{insight['severity']}</span>
                    <span style='color: white;'>{insight['status']}</span>
                </div>
            </div>
            <div style='color: var(--light-text-secondary);'>{insight['description']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent Security Events
    st.markdown("#### 📋 Recent Security Events")
    
    events = pd.DataFrame({
        'Time': ['10:30 AM', '10:15 AM', '09:45 AM', '09:20 AM', '08:50 AM'],
        'Event': ['Jailbreak attempt detected', 'Prompt injection blocked', 'Data leakage prevented', 
                  'Adversarial attack mitigated', 'Unauthorized access attempt'],
        'Model': ['GPT-4', 'Claude 3', 'Llama 3', 'GPT-4', 'Gemini Pro'],
        'Severity': ['Critical', 'High', 'Medium', 'High', 'Low'],
        'Status': ['Blocked', 'Blocked', 'Prevented', 'Mitigated', 'Investigated']
    })
    
    st.dataframe(events, use_container_width=True, hide_index=True)

# Footer
st.divider()
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"**EMVO Agentic AI Platform** • Based on Beam AI Agent Templates Architecture • {current_time}")
st.markdown("Enterprise AI agent management with observability, fine-tuning, and security evaluation capabilities")
