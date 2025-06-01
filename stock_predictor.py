# Third Party
import os
import torch
import twstock
import pandas as pd
from transformers import TextStreamer   
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from twstock import Stock, BestFourPoint
from datasets import Dataset as HFDataset
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_sharegpt
from unsloth.chat_templates import train_on_responses_only
from transformers import TrainingArguments, DataCollatorForSeq2Seq
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# Local
from utils import *
from dataset import StockDataset


class StockPredictor:
    def __init__(self, args):
        self.args = args

    def get_realtime_tw_stock(self, stock_code):
        return twstock.realtime.get(stock_code)

    def choose_stock(self, num = 10):        
        company_list = []
        info_list = [info for code, info in twstock.codes.items() if info[5] == '上市']
        buy_num = num
        sell_num = num
        for info in info_list:
            try:
                stock = Stock(str(info[1]))
                bfp = BestFourPoint(stock)
            except:
                continue
            
            if(bfp.best_four_point_to_buy() and buy_num > 0):      
                company_list.append({'name_zh':info[2], 'code':info[1]})            
                buy_num -= 1

            if(bfp.best_four_point_to_sell() and sell_num > 0): 
                company_list.append({'name_zh':info[2], 'code':info[1]})
                sell_num -= 1

            if(buy_num == 0 and sell_num == 0):
                break

        return company_list
           
    def analyze_stock_code_yearly(self, stock_code, year, dir = './'):
        data_columns = ['stock_code', 'date', ]
        analyzed_stock_data = []

        self.choose_stock()

        os.makedirs(dir)
        filename = f'{stock_code}_{year}.csv'
        file_path = os.path.join(dir, filename)
        pd.DataFrame(analyzed_stock_data).to_csv(file_path)

    def train(self):
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

        # Prepare
        self.load_pretrained()
        self.update_data()
        self.collect_data()
        self.get_trainer()

        # GPU State
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        # Train
        print("Training...")
        trainer_stats = self.trainer.train()
        print("Training done!")

        # Save model
        self.save_model()

        # Show final memory and time stats
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory         /max_memory*100, 3)
        lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


    def load_pretrained(self):
        print("Loading pretrained model...")
        # Load pretrained model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.args.model_name,
            max_seq_length = self.args.max_seq_length,
            dtype = None,
            load_in_4bit = self.args.load_in_4bit,
            load_in_8bit = self.args.load_in_8bit,
        )

        # Apply Chat Template
        self.tokenizer = get_chat_template(
            tokenizer,
            chat_template = "llama-3.1",
        )

        # LoRA
        self.model = FastLanguageModel.get_peft_model(
                    model,
                    r = self.args.rank,
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
                    lora_alpha = self.args.lora_alpha,
                    lora_dropout = self.args.lora_dropout,
                    bias = self.args.bias, 
                    use_gradient_checkpointing = "unsloth",
                    random_state = self.args.random_state,
                    use_rslora = False,  
                    loftq_config = None,
                )
        
    def fetch_stock_data(self, companies):
        company_no_codes = []
        for company in companies:
            if 'code' in company:
                try:
                    stock = Stock(company["code"])
                    datas = stock.fetch_from(self.args.year, self.args.month)
                    df = pd.DataFrame(datas)
                    df.to_csv(os.path.join(self.args.data_dir, 'stocks', f'{company['name_zh']}_{company["code"]}_stocks.csv'))
                except:
                    print(f'No data for {company["name_zh"]}')
                    company_no_codes.append(company)
                    continue
            else:
                company_no_codes.append(company)

        # company_no_codes_df = pd.DataFrame(company_no_codes)
        # company_no_codes_df.to_csv(os.path.join(self.args.data_dir, 'company_no_codes.csv'))
    
    def update_data(self):
        print("Updating data...")
        
        # Check file
        list_file = os.path.join(self.args.data_dir, self.args.company_list_file)
        stocks_dir = os.path.join(self.args.data_dir, 'stocks')
        os.makedirs(stocks_dir, exist_ok=True)

        if (os.path.exists(list_file)):
            companies = json.load(open(list_file))
            for company in companies:
                name_zh = company["name_zh"]
                for code, stock_info in twstock.codes.items():
                    if name_zh == stock_info.name:
                        company["code"] = code

            self.fetch_stock_data(companies)            
        else:
            print(f"No file with target company list found, Randomly choose {self.args.choose_stock} stocks to train")
            companies = self.choose_stock(self.args.choose_stock)
            print("Collected companies: ", companies)
            self.fetch_stock_data(companies)
        
    
    def collect_data(self): 
        print("Collecting data...")
        def formatting_prompts_func(examples):
            convos = examples["conversations"]
            texts = [self.tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
            return { "text" : texts}         

        def convert_to_hf_dataset(stock_dataset):
            hf_data = []
            for conversation_pair in stock_dataset.data:
                hf_data.append({"conversations": conversation_pair})
            return HFDataset.from_list(hf_data)
        
        # Load dataset
        self.dataset = StockDataset(self.args.data_dir + '/stocks')    

        # Format dataset
        hf_dataset = convert_to_hf_dataset(self.dataset)
        hf_dataset = standardize_sharegpt(hf_dataset)
        self.hf_dataset = hf_dataset.map(formatting_prompts_func, batched = True)
        
        print("Training Dataset:", self.hf_dataset)

    def get_trainer(self):
        print("Getting trainer...")
        trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = self.hf_dataset,
            # dataset_text_field = "text",
            # max_seq_length = self.args.max_seq_length,
            data_collator = DataCollatorForSeq2Seq(tokenizer = self.tokenizer),
            # dataset_num_proc = 2,
            # packing = False, 
            # args = TrainingArguments(
            args = SFTConfig(
                max_seq_length = self.args.max_seq_length,
                dataset_text_field = "text",
                dataset_num_proc = 2,
                packing = False, 
                per_device_train_batch_size = self.args.batch_size,
                gradient_accumulation_steps = self.args.gradient_accumulation_steps,
                warmup_steps = self.args.warmup_steps,
                num_train_epochs = self.args.num_epochs,
                max_steps = self.args.max_steps,
                learning_rate = self.args.learning_rate,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = self.args.logging_steps,
                optim = self.args.optim,
                weight_decay = self.args.weight_decay,
                lr_scheduler_type = self.args.lr_scheduler_type,
                seed = self.args.random_state,
                output_dir = self.args.output_dir,
                report_to = self.args.report_to,
            ),
        )

        self.trainer = train_on_responses_only(
            trainer,
            instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
            response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )

    def save_model(self):        
        print("Saving model...")
        self.model.save_pretrained(self.args.model_dir) 
        self.tokenizer.save_pretrained(self.args.model_dir)

    def load_model(self):
        print("Loading finetuned model...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.args.model_dir,
            dtype = None,
            load_in_4bit = self.args.load_in_4bit,
            load_in_8bit = self.args.load_in_8bit,
        )
        FastLanguageModel.for_inference(self.model)

    def inference(self, code='2330', past=30, future=30, context=""):
        print("Starting inference...")         
        self.load_model()

        today = datetime.today()
        past_n_work_date = get_workdays(today, past, direction='past')
        future_work_date = get_workdays(today, future, direction='future')

        company_name = search_company_name_code(code)
        
        stock_infos = fetch_twse_data(code, past_n_work_date, past)
        user_content = f"Here is past {past} day stock price of {company_name} company.\n\n"
        for info in stock_infos:
            for field in self.args.fields:
                if field in info:
                    user_content += f'{field}: {info[field]}, '
            user_content += '\n'
        user_content += f'Predict the next {future} day price:\n\n'
        
        message = [
            {
                'role': 'system', 
                'content': 'You are a stock trader. You are given a company name and a list of dates. You need to predict the next n days price of the company.' + context
            },
            {
                'role': 'user',
                'content': user_content
            }
        ]

        result_df = self.generate(message)        

        self.plot_kline_graph(result_df, company_name, code)

    def generate(self, message):
        print("Generating response...")    
        inputs = self.tokenizer.apply_chat_template(
            message,
            tokenize = True,
            add_generation_prompt = True, # Must add for generation
            return_tensors = "pt",
        ).to("cuda")

        text_streamer = TextStreamer(self.tokenizer, skip_prompt = True)
        outputs = self.model.generate(input_ids = inputs, streamer = text_streamer, #`max_new_tokens = 128,
                        use_cache = True, temperature = 1.5, min_p = 0.1)
              
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)        
        organized_data = parse_prediction_output(generated_text, self.args.fields)
        result_df = pd.DataFrame(organized_data)

        return result_df

    def plot_kline_graph(self, df, company_name, code):
        # Calculate change
        pre_close = df['close'].iloc[0]
        for idx, row in df.iterrows():
            if pd.isna(row['change']):
                df.loc[idx, 'change'] = row['close'] - pre_close
            pre_close = row['close']
        # pre_close = df['close'].iloc[0]
        # for _, row in df.iterrows():
        #     if row['change'].isna():
        #         row['change'] = row['close'] - pre_close
        #     pre_close = row['close']

        # df['computed_change'] = df['close'].diff()
        # df['change'] = df['change'].fillna(df['computed_change'])
        # df.drop(columns='computed_change', inplace=True)
        

        # Moving Averages
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA60'] = df['close'].rolling(window=60).mean()

        # 建立子圖：價格圖在上，成交量在下
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            # subplot_titles=('股價', ''),
            row_width=[0.2, 0.7]  # 價格圖占70%，成交量圖占30%
        )

        # 自訂 K 線 hovertext
        hover_texts = [
            f"<b>日期:</b> {row['date']}<br>"
            f"<b>開:</b> {row['open']:.2f}<br>"
            f"<b>高:</b> {row['high']:.2f}<br>"
            f"<b>低:</b> {row['low']:.2f}<br>"
            f"<b>收:</b> {row['close']:.2f}<br>"
            f"<b>漲跌:</b> {row['change']:.2f}<br>"
            f"<b>成交量:</b> {row['transaction']:,}"
            for _, row in df.iterrows()
        ]

        red = "#ED3E3E"
        green = "#42A875"

        # 第一行：加入 K 線圖（設定漲跌顏色）
        fig.add_trace(go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K線圖',
            hovertext=hover_texts,
            hoverinfo='text',
            increasing_line_color= red,      # 上漲K線邊框顏色
            increasing_fillcolor=red,       # 上漲K線填充顏色
            decreasing_line_color=green,    # 下跌K線邊框顏色
            decreasing_fillcolor=green      # 下跌K線填充顏色
        ), row=1, col=1)

        # 第一行：加入 MA 線
        fig.add_trace(go.Scatter(
            x=df['date'], 
            y=df['MA5'],  
            name='MA5',  
            line=dict(color='blue', width=1),
            mode='lines',
            hovertemplate='<b>MA5:</b> %{y:.2f}<extra></extra>'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df['date'], 
            y=df['MA20'], 
            name='MA20', 
            line=dict(color='purple', width=1),
            mode='lines',
            hovertemplate='<b>MA20:</b> %{y:.2f}<extra></extra>'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df['date'], 
            y=df['MA60'], 
            name='MA60', 
            line=dict(color='orange', width=1),
            mode='lines',
            hovertemplate='<b>MA60:</b> %{y:.2f}<extra></extra>'
        ), row=1, col=1)

        # 第二行：加入成交量柱狀圖
        # 根據漲跌決定顏色
        colors = [red if change >= 0 else green for change in df['change']]

        fig.add_trace(go.Bar(
            x=df['date'],
            y=df['transaction'],
            name='成交量',
            showlegend=False,
            marker_color=colors,
            hovertemplate='<b>日期:</b> %{x}<br><b>成交量:</b> %{y:,}<extra></extra>'
        ), row=2, col=1)

        # 格式設定
        fig.update_layout(
            title=f'{company_name} ({code}) 股價走勢圖',
            # xaxis_rangeslider_visible=False,
            template='plotly_white',
            hovermode='x unified',
            height=700  # 增加圖表高度
        )

        # 設定 y 軸標題
        # fig.update_yaxes(title_text="價格", row=1, col=1)
        # fig.update_yaxes(title_text="成交量", row=2, col=1)

        # 隱藏 x 軸標題（因為下方的成交量圖會顯示）
        fig.update_xaxes(title_text="", row=1, col=1)
        fig.update_xaxes(title_text="日期", row=2, col=1)

        fig.write_html(f"{company_name}_K線圖.html")



    def plot_compare_graph(self, past=30, future=10, context=""):         
        self.load_model()
        data_path = os.path.join(self.args.data_dir, 'stocks')

        for file in os.listdir(data_path):
            if file.endswith('.csv'):                
                company = file.split('_')[0]
                df = pd.read_csv(os.path.join(data_path, file)).tail(past + future)                
                print(f"Processing {company} data...")
                try:
                    user_content = f"Here is past {past} day stock price of {company} company.\n\n"
                    for _, row in df[:past].iterrows():
                        for field in self.args.fields:
                            if field in row:
                                user_content += f'{field}: {row[field]}, '
                        user_content += '\n'
                    user_content += f'Predict the next {future} day price:\n\n'
                    
                    message = [
                        {
                            'role': 'system', 
                            'content': 'You are a stock trader. You are given a company name and a list of dates. You need to predict the next n days price of the company.' + context
                        },
                        {
                            'role': 'user',
                            'content': user_content
                        }
                    ]

                    df_pred = self.generate(message)

                    if(len(df_pred) > 0):                
                        title = f"{company} Stock Forecast vs Actual: {past} Days Input → {future} Days Prediction"        
                        self.plot_compare_line(df, df_pred, company, label='Close Price', title=title, y ='close', x='date')
                        self.plot_compare_line(df, df_pred, company, label='Open Price', title=title, y ='open', x='date')
                        self.plot_compare_line(df, df_pred, company, label='Transaction Price', title=title, y ='transaction', x='date')
                except Exception as e:
                    print(f"Error processing {company}: {e}")
                    continue
    
    # def plot_compare_line(self, df_ori, df_pred, company, label, y='close', x='date'):        
    #     print('Plotting compared line for price...')
    #     try:
    #         plt.figure(figsize=(20, 7))
    #         plt.plot(df_ori[x], df_ori[y], label = f'{label} (Original)')
    #         plt.plot(df_pred[x], df_pred[y], label = f'{label} (Predicted)')
    #         plt.title(f"{company} Stock Forecast vs Actual: 30 Days Input → 10 Days Prediction")
    #         plt.xlabel('Date')
    #         plt.ylabel('Price')
    #         plt.legend()
    #         plt.grid()
    #         os.makedirs('result', exist_ok=True)
    #         plt.savefig(f"result/{company}_{label}_price.png")
    #     except Exception as e:
    #         print(f"Error plotting {label} for {company}: {e}")
    #         return


    def plot_compare_line(self, df_ori, df_pred, company, label, title, y='close', x='date'):
        print('Plotting compared line for price...')
        try:
            fig = go.Figure()

            # 原始資料線
            fig.add_trace(go.Scatter(
                x=df_ori[x],
                y=df_ori[y],
                mode='lines',
                name=f'{label} (Original)',
                line=dict(color='blue')
            ))

            # 預測資料線
            fig.add_trace(go.Scatter(
                x=df_pred[x],
                y=df_pred[y],
                mode='lines',
                name=f'{label} (Predicted)',
                line=dict(color='red', dash='dash')
            ))

            fig.update_layout(
                title=title,
                xaxis_title='Date',
                yaxis_title='Price',
                legend=dict(
                            x=1,
                            y=1.4,
                            xanchor='right',
                            yanchor='top'
                        ),
                template='plotly_white',
                width=1200,
                height=500
            )

            os.makedirs('result', exist_ok=True)
            fig.write_image(f"result/{company}_{label}_price.png", engine='kaleido')

        except Exception as e:
            print(f"Error plotting {label} for {company}: {e}")
            return

    
    