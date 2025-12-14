import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import json

# Page configuration
st.set_page_config(
    page_title="EMVO Agentic AI Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 10px;
    }
    .section-title {
        font-size: 1.8rem;
        color: #2563EB;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3B82F6;
    }
    .subsection-title {
        font-size: 1.4rem;
        color: #4B5563;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .agent-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        transition: transform 0.2s;
    }
    .agent-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px -2px rgba(0,0,0,0.08);
    }
    .chat-bubble {
        background: #F3F4F6;
        padding: 0.8rem 1.2rem;
        border-radius: 18px;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    .user-bubble {
        background: #3B82F6;
        color: white;
        margin-left: auto;
    }
    .agent-bubble {
        background: #E5E7EB;
        color: #1F2937;
    }
    .tab-content {
        padding: 1.5rem 0;
    }
    .success-badge {
        background-color: #10B981;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        display: inline-block;
    }
    .warning-badge {
        background-color: #F59E0B;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        display: inline-block;
    }
    .error-badge {
        background-color: #EF4444;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history and agent selection
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}
if 'selected_agent' not in st.session_state:
    st.session_state.selected_agent = None
if 'active_agent_tab' not in st.session_state:
    st.session_state.active_agent_tab = "chat"

# Sample data for agents (in production, this would come from a database)
def initialize_agent_data():
    agents = {
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
            }
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
            }
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
            }
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
            }
        }
    }
    return agents

# Title
st.markdown('<h1 class="main-title">🤖 EMVO Agentic AI Platform</h1>', unsafe_allow_html=True)
st.markdown("### Production-Ready AI Agent Management & Monitoring System")

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Select Dashboard:",
    ["📊 Dashboard", "🤖 Agentverse", "🧠 Brains", "⚖️ Judge"]
)

# Dashboard Section
if section == "📊 Dashboard":
    st.markdown('<h2 class="section-title">📊 Agent Observatory Dashboard</h2>', unsafe_allow_html=True)
    
    agents = initialize_agent_data()
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_agents = len(agents)
    active_agents = sum(1 for a in agents.values() if a['status'] == 'active')
    total_conversations = sum(a['conversations'] for a in agents.values())
    avg_success_rate = np.mean([a['success_rate'] for a in agents.values()])
    
    with col1:
        st.metric("Total Agents", total_agents, f"{active_agents} active")
    with col2:
        st.metric("Total Conversations", f"{total_conversations:,}")
    with col3:
        st.metric("Avg Success Rate", f"{avg_success_rate:.1f}%")
    with col4:
        avg_response = np.mean([a['avg_response_time'] for a in agents.values()])
        st.metric("Avg Response Time", f"{avg_response:.1f}s")
    
    st.markdown("---")
    
    # Agent Status Overview
    st.markdown('<h3 class="subsection-title">Agent Status Overview</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create agent status chart
        agent_names = list(agents.keys())
        status_colors = {'active': '#10B981', 'warning': '#F59E0B', 'error': '#EF4444'}
        status_values = [agents[name]['status'] for name in agent_names]
        
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
                      line=dict(color='#3B82F6', width=3))
        ])
        
        fig.update_layout(
            title='Agent Performance Metrics',
            yaxis=dict(title='Total Conversations'),
            yaxis2=dict(title='Success Rate (%)', overlaying='y', side='right'),
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Agent Status")
        for agent_name, agent_data in agents.items():
            status_badge = f"<span class='{agent_data['status']}-badge'>{agent_data['status'].upper()}</span>"
            st.markdown(f"""
            <div class='agent-card'>
                <strong>{agent_name}</strong><br>
                {status_badge}<br>
                <small>Type: {agent_data['type']}</small><br>
                <small>Template: {agent_data['template']}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Recent Activity Timeline
    st.markdown('<h3 class="subsection-title">Recent Agent Activity</h3>', unsafe_allow_html=True)
    
    # Generate timeline data
    timeline_data = []
    for agent_name, agent_data in agents.items():
        for chat in agent_data['chat_history'][:2]:  # Last 2 chats per agent
            timeline_data.append({
                'agent': agent_name,
                'message': chat['user'][:50] + "..." if len(chat['user']) > 50 else chat['user'],
                'time': chat['time'],
                'type': 'user_query'
            })
            timeline_data.append({
                'agent': agent_name,
                'message': chat['agent'][:50] + "..." if len(chat['agent']) > 50 else chat['agent'],
                'time': chat['time'],
                'type': 'agent_response'
            })
    
    timeline_df = pd.DataFrame(timeline_data)
    if not timeline_df.empty:
        st.dataframe(timeline_df, use_container_width=True, hide_index=True)

# Agentverse Section
elif section == "🤖 Agentverse":
    st.markdown('<h2 class="section-title">🤖 EMVO Agentverse</h2>', unsafe_allow_html=True)
    
    agents = initialize_agent_data()
    
    # Agent selection
    st.markdown('<h3 class="subsection-title">Select an Agent</h3>', unsafe_allow_html=True)
    
    cols = st.columns(4)
    for idx, (agent_name, agent_data) in enumerate(agents.items()):
        with cols[idx % 4]:
            if st.button(f"**{agent_name}**\n\n{agent_data['type']}\nStatus: {agent_data['status']}", 
                        key=f"select_{agent_name}",
                        use_container_width=True):
                st.session_state.selected_agent = agent_name
                st.rerun()
    
    st.markdown("---")
    
    # Selected agent interface
    if st.session_state.selected_agent:
        agent_name = st.session_state.selected_agent
        agent_data = agents[agent_name]
        
        st.markdown(f'<h3 class="subsection-title">🎯 {agent_name} - Interactive Interface</h3>', unsafe_allow_html=True)
        
        # Tabs for different interfaces
        tab1, tab2, tab3 = st.tabs(["💬 Chat Interface", "🎤 Voice Interface", "📊 Agent Analytics"])
        
        with tab1:
            st.markdown(f"**Template:** {agent_data['template']} | **Status:** {agent_data['status']}")
            
            # Chat container
            chat_container = st.container(height=400)
            
            with chat_container:
                for chat in agent_data['chat_history']:
                    st.markdown(f"""
                    <div class="chat-bubble user-bubble">
                        <strong>User ({chat['time']}):</strong><br>
                        {chat['user']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="chat-bubble agent-bubble">
                        <strong>{agent_name}:</strong><br>
                        {chat['agent']}
                    </div>
                    """, unsafe_allow_html=True)
            
            # New message input
            col1, col2 = st.columns([4, 1])
            with col1:
                new_message = st.text_input("Type your message:", key="new_message")
            with col2:
                if st.button("Send", use_container_width=True):
                    if new_message:
                        # In production, this would call the actual agent API
                        st.success(f"Message sent to {agent_name}!")
                        st.rerun()
        
        with tab2:
            st.markdown("### Voice Interface")
            st.info("Voice interface integrates with Beam AI's agentic patterns for real-time voice support")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Voice Controls")
                if st.button("🎤 Start Voice Session", use_container_width=True):
                    st.success("Voice session started with agent!")
                
                if st.button("⏸️ Mute Voice", use_container_width=True):
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
                st.metric("Success Rate", f"{agent_data['success_rate']}%")
            with col2:
                st.metric("Avg Response Time", f"{agent_data['avg_response_time']}s")
            with col3:
                st.metric("Total Conversations", agent_data['conversations'])
            
            # Analytics charts
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                y=agent_data['analytics']['daily_interactions'],
                mode='lines+markers',
                name='Daily Interactions',
                line=dict(color='#3B82F6', width=3)
            ))
            fig1.update_layout(
                title='Weekly Interaction Trends',
                height=300
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
                yaxis=dict(range=[0, 5])
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    else:
        st.info("👆 Select an agent from above to interact with it")

# Brains Section
elif section == "🧠 Brains":
    st.markdown('<h2 class="section-title">🧠 EMVO Brains - Auto LLM Platform</h2>', unsafe_allow_html=True)
    
    st.markdown("### Automated LLM Fine-Tuning & Optimization")
    
    # Search for LLM models/platforms
    st.markdown('<h3 class="subsection-title">🔍 Search for Auto LLM Platforms</h3>', unsafe_allow_html=True)
    
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        platform_search = st.text_input("Search platforms or models:", placeholder="e.g., GPT-4, Llama 3, Claude, fine-tuning...")
    with search_col2:
        search_btn = st.button("Search", use_container_width=True)
    
    # Available platforms (simulated data)
    platforms = [
        {"name": "OpenAI Fine-Tuning", "type": "Commercial", "max_context": "128K", "best_for": "General purpose, high accuracy"},
        {"name": "Anthropic Claude", "type": "Commercial", "max_context": "200K", "best_for": "Safety, long documents"},
        {"name": "Llama 3 70B", "type": "Open Source", "max_context": "8K", "best_for": "Customization, on-premise"},
        {"name": "Gemini Pro", "type": "Commercial", "max_context": "1M", "best_for": "Multimodal tasks"},
        {"name": "Mistral Large", "type": "Commercial", "max_context": "32K", "best_for": "European languages, efficiency"}
    ]
    
    filtered_platforms = [p for p in platforms if platform_search.lower() in p['name'].lower()] if platform_search else platforms
    
    st.dataframe(pd.DataFrame(filtered_platforms), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Fine-tuning configuration
    st.markdown('<h3 class="subsection-title">⚙️ Fine-Tuning Configuration</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Hyperparameters")
        
        selected_model = st.selectbox("Base Model:", ["GPT-4", "Llama 3", "Claude 3", "Custom Model"])
        
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
        st.markdown("#### Training Data Configuration")
        
        dataset_type = st.radio("Dataset Type:", ["Structured", "Unstructured", "Mixed"])
        
        if dataset_type == "Structured":
            st.number_input("Number of examples:", 100, 100000, 1000)
            st.text_area("Data schema:", value="{\"prompt\": \"\", \"completion\": \"\"}", height=100)
        elif dataset_type == "Unstructured":
            uploaded_files = st.file_uploader("Upload training files:", 
                                            accept_multiple_files=True,
                                            type=['txt', 'json', 'csv', 'pdf'])
            if uploaded_files:
                st.write(f"{len(uploaded_files)} files uploaded")
        else:
            st.number_input("Structured examples:", 100, 50000, 1000)
            uploaded_files = st.file_uploader("Upload unstructured files:", 
                                            accept_multiple_files=True,
                                            type=['txt', 'pdf', 'docx'])
        
        validation_split = st.slider("Validation Split %", 5, 40, 20)
    
    # Training performance section
    st.markdown("---")
    st.markdown('<h3 class="subsection-title">📈 Training Performance Monitor</h3>', unsafe_allow_html=True)
    
    # Simulated training metrics
    epochs_list = list(range(1, epochs + 1))
    train_loss = [2.5 - (2.3 * (e-1)/(epochs-1)) + np.random.normal(0, 0.1) for e in epochs_list]
    val_loss = [2.4 - (2.2 * (e-1)/(epochs-1)) + np.random.normal(0, 0.15) for e in epochs_list]
    accuracy = [0.65 + (0.25 * (e-1)/(epochs-1)) + np.random.normal(0, 0.05) for e in epochs_list]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs_list, y=train_loss, mode='lines+markers', name='Training Loss', line=dict(color='#3B82F6')))
    fig.add_trace(go.Scatter(x=epochs_list, y=val_loss, mode='lines+markers', name='Validation Loss', line=dict(color='#EF4444')))
    fig.add_trace(go.Scatter(x=epochs_list, y=accuracy, mode='lines+markers', name='Accuracy', 
                            yaxis='y2', line=dict(color='#10B981')))
    
    fig.update_layout(
        title='Training Progress',
        xaxis=dict(title='Epoch'),
        yaxis=dict(title='Loss'),
        yaxis2=dict(title='Accuracy', overlaying='y', side='right'),
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Start training button
    if st.button("🚀 Start Fine-Tuning", type="primary", use_container_width=True):
        st.success("Fine-tuning job started! Check back for progress updates.")
        # In production, this would trigger actual training

# Judge Section
elif section == "⚖️ Judge":
    st.markdown('<h2 class="section-title">⚖️ EMVO Judge - LLM as a Judge Research</h2>', unsafe_allow_html=True)
    
    st.markdown("### Research Platform for LLM Evaluation & Benchmarking")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Evaluation Parameters")
        
        evaluation_type = st.selectbox("Evaluation Type:", 
                                      ["Safety & Alignment", "Factual Accuracy", "Reasoning Ability", 
                                       "Bias Detection", "Adversarial Robustness"])
        
        st.markdown("**Test Categories:**")
        col1a, col1b = st.columns(2)
        with col1a:
            st.checkbox("Jailbreak attempts", True)
            st.checkbox("Prompt injection", True)
            st.checkbox("Harmful content", True)
        with col1b:
            st.checkbox("Privacy leaks", True)
            st.checkbox("Misinformation", True)
            st.checkbox("Bias detection", True)
        
        test_intensity = st.slider("Test Intensity:", 1, 10, 7)
        num_test_cases = st.number_input("Number of test cases:", 10, 10000, 100)
    
    with col2:
        st.markdown("#### Model Comparison")
        
        models_to_compare = st.multiselect("Select models to compare:",
                                          ["GPT-4", "Claude 3", "Llama 3", "Gemini Pro", "Mistral Large"],
                                          default=["GPT-4", "Claude 3"])
        
        benchmark_metrics = st.multiselect("Benchmark metrics:",
                                          ["Truthfulness", "Safety", "Helpfulness", "Reasoning", 
                                           "Creativity", "Efficiency", "Alignment"],
                                          default=["Truthfulness", "Safety", "Helpfulness"])
    
    st.markdown("---")
    
    # Research results visualization
    st.markdown('<h3 class="subsection-title">📊 Research Results & Analysis</h3>', unsafe_allow_html=True)
    
    # Simulated research data
    if models_to_compare:
        metrics_data = []
        for model in models_to_compare:
            for metric in benchmark_metrics:
                score = np.random.uniform(0.7, 0.95)  # Simulated scores
                metrics_data.append({
                    'Model': model,
                    'Metric': metric,
                    'Score': score,
                    'Status': 'Pass' if score > 0.8 else 'Needs Review'
                })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Heatmap visualization
        pivot_df = df_metrics.pivot(index='Model', columns='Metric', values='Score')
        
        fig = px.imshow(pivot_df,
                       text_auto='.2f',
                       aspect='auto',
                       color_continuous_scale='RdYlGn',
                       title=f"LLM Evaluation Results - {evaluation_type}")
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed findings
        st.markdown("#### Detailed Findings")
        
        findings = {
            "Safety & Alignment": "All models show strong alignment with human values. Claude 3 excels in safety filtering.",
            "Factual Accuracy": "GPT-4 leads in factual accuracy across diverse domains. Open-source models need improvement.",
            "Reasoning Ability": "Complex reasoning tasks show variance. Chain-of-thought prompting improves performance.",
            "Adversarial Robustness": "All models vulnerable to sophisticated jailbreaks. Continuous monitoring required."
        }
        
        for category, finding in findings.items():
            if evaluation_type == category or evaluation_type == "All":
                with st.expander(f"🔍 {category}"):
                    st.write(finding)
                    st.metric("Average Score", f"{np.random.uniform(0.75, 0.92):.2%}")
        
        # Research controls
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔬 Run New Evaluation", use_container_width=True):
                st.success(f"Starting {evaluation_type} evaluation on {len(models_to_compare)} models...")
        with col2:
            if st.button("📥 Export Research Data", use_container_width=True):
                st.info("Research data exported to CSV format")
        with col3:
            if st.button("📈 Generate Report", use_container_width=True):
                st.info("Comprehensive research report generated")
    else:
        st.warning("Please select at least one model to compare")

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem; padding: 1rem;">
    <p>EMVO Agentic AI Platform • Based on Beam AI Agent Templates Architecture • {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p>Production-ready AI agent management with observability, fine-tuning, and evaluation capabilities</p>
</div>
""", unsafe_allow_html=True)