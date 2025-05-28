import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def fetch_financial_data(ticker_symbol):
    """
    Fetches financial data (income statement, cash flow, balance sheet, and general info)
    for a given stock ticker using yfinance.
    """
    stock = yf.Ticker(ticker_symbol)
    income_statement = stock.financials
    cash_flow = stock.cashflow
    balance_sheet = stock.balance_sheet
    info = stock.info
    return income_statement, cash_flow, balance_sheet, info

def prepare_data(cash_flow, income_statement, balance_sheet):
    """
    Prepares and extracts key financial metrics from the raw data.
    """
    # Transpose cash flow for easier access
    cash_flow_transposed = cash_flow.transpose()
    
    # Handle potential missing 'Free Cash Flow'
    latest_free_cash_flow = cash_flow_transposed['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in cash_flow_transposed.columns and not cash_flow_transposed['Free Cash Flow'].empty else 0

    # Ensure there are enough data points for growth calculation
    if 'Net Income' in income_statement.index and len(income_statement.loc['Net Income']) > 1:
        latest_net_income = income_statement.loc['Net Income'].iloc[0]
        previous_net_income = income_statement.loc['Net Income'].iloc[1]
        if previous_net_income != 0: # Avoid division by zero
            net_income_growth_rate = (latest_net_income - previous_net_income) / abs(previous_net_income)
        else:
            net_income_growth_rate = 0 
    else:
        latest_net_income = income_statement.loc['Net Income'].iloc[0] if 'Net Income' in income_statement.index and not income_statement.loc['Net Income'].empty else 0
        net_income_growth_rate = 0 # Default if not enough data

    if 'Total Revenue' in income_statement.index and len(income_statement.loc['Total Revenue']) > 1:
        latest_revenue = income_statement.loc['Total Revenue'].iloc[0]
        previous_revenue = income_statement.loc['Total Revenue'].iloc[1]
        if previous_revenue != 0: # Avoid division by zero
            revenue_growth_rate = (latest_revenue - previous_revenue) / abs(previous_revenue)
        else:
            revenue_growth_rate = 0
    else:
        latest_revenue = income_statement.loc['Total Revenue'].iloc[0] if 'Total Revenue' in income_statement.index and not income_statement.loc['Total Revenue'].empty else 0
        revenue_growth_rate = 0 # Default if not enough data
    
    latest_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index and not balance_sheet.loc['Total Assets'].empty else 0
    
    if 'Total Debt' in balance_sheet.index and not balance_sheet.loc['Total Debt'].empty:
        latest_liabilities = balance_sheet.loc['Total Debt'].iloc[0]
    elif 'Long Term Debt' in balance_sheet.index and not balance_sheet.loc['Long Term Debt'].empty:
        latest_liabilities = balance_sheet.loc['Long Term Debt'].iloc[0]
        if 'Current Debt' in balance_sheet.index and not balance_sheet.loc['Current Debt'].empty and pd.notna(balance_sheet.loc['Current Debt'].iloc[0]):
             latest_liabilities += balance_sheet.loc['Current Debt'].iloc[0]
    else:
        latest_liabilities = 0 

    equity = latest_assets - latest_liabilities
    
    return latest_free_cash_flow, net_income_growth_rate, revenue_growth_rate, equity, latest_assets, latest_liabilities, latest_revenue, latest_net_income

def calculate_dcf(free_cash_flow, discount_rate, growth_rate, years=5):
    """
    Calculates Discounted Cash Flow (DCF) value and future cash flows.
    """
    projected_cash_flows_discounted = []
    future_cash_flows_nominal = []
    
    growth_rate = min(growth_rate, 0.5) if growth_rate > 0 else max(growth_rate, -0.5) # Cap growth rate

    for year in range(1, years + 1):
        future_cash_flow_nominal_val = free_cash_flow * ((1 + growth_rate) ** year)  
        discounted_cash_flow_val = future_cash_flow_nominal_val / ((1 + discount_rate) ** year)  
        projected_cash_flows_discounted.append(discounted_cash_flow_val)
        future_cash_flows_nominal.append(future_cash_flow_nominal_val)

    dcf_value = sum(projected_cash_flows_discounted)
    return dcf_value, projected_cash_flows_discounted, future_cash_flows_nominal

def calculate_pe_ratio(latest_net_income, shares_outstanding, current_price):
    """
    Calculates the Price-to-Earnings (P/E) ratio.
    """
    if shares_outstanding is None or shares_outstanding == 0:
        print("Shares outstanding is zero or not available, cannot calculate P/E ratio.")
        return None
    if latest_net_income <= 0:
        print(f"Net income is {latest_net_income:,.2f}. P/E ratio is not meaningful.")
        return None 
        
    eps = latest_net_income / shares_outstanding
    if eps == 0: 
        print("EPS is zero, P/E ratio is undefined.")
        return None
    pe_ratio = current_price / eps
    return pe_ratio

def calculate_future_stock_prices(latest_price, growth_rate, years):
    """
    Projects future stock prices based on a constant growth rate.
    """
    future_prices = []
    growth_rate = min(growth_rate, 0.5) if growth_rate > 0 else max(growth_rate, -0.5) # Cap growth rate

    for year in range(1, years + 1):
        future_price = latest_price * ((1 + growth_rate) ** year)  
        future_prices.append(future_price)
    return future_prices

def analyze_financials(income_statement, cash_flow, balance_sheet, info, discount_rate=0.1, ebitda_multiple=10):
    """
    Performs a more detailed financial analysis and prints key ratios.
    """
    print("\n--- Detailed Financial Analysis ---")
    
    total_revenue = income_statement.loc['Total Revenue'] if 'Total Revenue' in income_statement.index else pd.Series([0], index=income_statement.columns if not income_statement.empty else [0])
    
    if 'Total Expenses' in income_statement.index:
        total_expenses = income_statement.loc['Total Expenses']
    else: 
        expense_items = ['Cost Of Revenue', 'Selling General And Administration', 'Research And Development', 'Operating Expense']
        total_expenses_calc = pd.Series([0.0] * len(income_statement.columns), index=income_statement.columns if not income_statement.empty else [0])
        for item in expense_items:
            if item in income_statement.index:
                total_expenses_calc += income_statement.loc[item].fillna(0)
        total_expenses = total_expenses_calc

    net_income = income_statement.loc['Net Income'] if 'Net Income' in income_statement.index else pd.Series([0], index=income_statement.columns if not income_statement.empty else [0])
    
    cash_flow_transposed = cash_flow.transpose()
    free_cash_flow_series = cash_flow_transposed['Free Cash Flow'] if 'Free Cash Flow' in cash_flow_transposed.columns else pd.Series([0], index=cash_flow_transposed.index if not cash_flow_transposed.empty else [0])
    
    ebitda = income_statement.loc['EBITDA'] if 'EBITDA' in income_statement.index else pd.Series([0], index=income_statement.columns if not income_statement.empty else [0])
    interest_expense = income_statement.loc['Interest Expense'] if 'Interest Expense' in income_statement.index else pd.Series([0], index=income_statement.columns if not income_statement.empty else [0])
    ebit = income_statement.loc['EBIT'] if 'EBIT' in income_statement.index else pd.Series([0], index=income_statement.columns if not income_statement.empty else [0])
    
    total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index and not balance_sheet.loc['Total Assets'].empty else 0
    
    if 'Total Debt' in balance_sheet.index and not balance_sheet.loc['Total Debt'].empty:
        total_debt_val = balance_sheet.loc['Total Debt'].iloc[0]
    elif 'Long Term Debt' in balance_sheet.index and not balance_sheet.loc['Long Term Debt'].empty:
        total_debt_val = balance_sheet.loc['Long Term Debt'].iloc[0]
        if 'Current Debt' in balance_sheet.index and not balance_sheet.loc['Current Debt'].empty and pd.notna(balance_sheet.loc['Current Debt'].iloc[0]):
            total_debt_val += balance_sheet.loc['Current Debt'].iloc[0]
    else:
        total_debt_val = 0

    shareholder_equity_val = balance_sheet.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in balance_sheet.index and not balance_sheet.loc['Stockholders Equity'].empty else (total_assets - total_debt_val)

    market_cap = info.get('marketCap', 0)
    latest_ebitda = ebitda.iloc[0] if not ebitda.empty else 0
    if latest_ebitda != 0 :
        enterprise_value = latest_ebitda * ebitda_multiple
        print(f"EBITDA-based valuation (EV/EBITDA of {ebitda_multiple}x) suggests an enterprise value of: ${enterprise_value:,.2f}")
    else:
        print("EBITDA is zero or unavailable, cannot calculate EBITDA-based enterprise value.")

    if 'Gross Profit' in income_statement.index:
        gross_profit = income_statement.loc['Gross Profit']
        if not total_revenue.empty and total_revenue.iloc[0] != 0:
            gross_margin = (gross_profit.iloc[0] / total_revenue.iloc[0]) * 100
            print(f"Gross Profit Margin (latest): {gross_margin:.2f}%")
        else:
            print("Total revenue is zero, cannot calculate Gross Profit Margin.")
    else:
        print("Gross Profit not available in income statement.")

    if not total_revenue.empty and total_revenue.iloc[0] != 0 and not ebit.empty:
        ebit_margin = (ebit.iloc[0] / total_revenue.iloc[0]) * 100
        print(f"EBIT Margin (latest): {ebit_margin:.2f}%")
    else:
        print("EBIT or Total Revenue is zero/unavailable, cannot calculate EBIT Margin.")

    if not interest_expense.empty and interest_expense.iloc[0] != 0 and not ebit.empty:
        interest_coverage = ebit.iloc[0] / interest_expense.iloc[0]
        print(f"Interest Coverage Ratio (latest): {interest_coverage:.2f}")
    elif not ebit.empty and (interest_expense.empty or interest_expense.iloc[0] == 0):
        print("No interest expenses for the period, or interest expense is zero. Interest coverage is very high/undefined.")
    else:
        print("EBIT or Interest Expense is unavailable for Interest Coverage Ratio.")

    current_price = info.get('currentPrice')
    shares_outstanding = info.get('sharesOutstanding')
    latest_net_income_val = net_income.iloc[0] if not net_income.empty else 0

    if current_price and shares_outstanding and latest_net_income_val > 0 :
        calculated_pe = calculate_pe_ratio(latest_net_income_val, shares_outstanding, current_price)
        if calculated_pe is not None:
            print(f"Calculated Price to Earnings (P/E) Ratio: {calculated_pe:.2f}")
        forward_pe = info.get('forwardPE')
        if forward_pe: print(f"Forward P/E Ratio (from yfinance): {forward_pe:.2f}")
        trailing_pe = info.get('trailingPE')
        if trailing_pe: print(f"Trailing P/E Ratio (from yfinance): {trailing_pe:.2f}")
    elif market_cap and latest_net_income_val > 0:
        pe_ratio_market_cap = market_cap / latest_net_income_val
        print(f"Price to Earnings (P/E) Ratio (using Market Cap): {pe_ratio_market_cap:.2f}")
    else:
        print("Current price, shares outstanding, or net income unavailable/non-positive for P/E calculation.")
        
    if shareholder_equity_val != 0:
        debt_to_equity = total_debt_val / shareholder_equity_val
        print(f"Debt to Equity Ratio: {debt_to_equity:.2f}")
        roe = (latest_net_income_val / shareholder_equity_val) * 100
        print(f"Return on Equity (ROE): {roe:.2f}%")
    else:
        print("Shareholder equity is zero, cannot calculate Debt to Equity or ROE.")

    if total_assets != 0:
        roa = (latest_net_income_val / total_assets) * 100
        print(f"Return on Assets (ROA): {roa:.2f}%")
    else:
        print("Total assets are zero, cannot calculate ROA.")

    invested_capital = total_debt_val + shareholder_equity_val
    if invested_capital != 0:
        roic_simple = (latest_net_income_val / invested_capital) * 100
        print(f"Return on Invested Capital (ROIC - simplified using Net Income): {roic_simple:.2f}%")
    else:
        print("Invested capital (Debt + Equity) is zero, cannot calculate ROIC.")

    latest_fcf = free_cash_flow_series.iloc[0] if not free_cash_flow_series.empty else 0
    if latest_fcf > 0 : 
        sustainable_growth_rate = 0.03 
        projected_cash_flows_dcf_list = []
        # This loop calculates discounted FCFs for the projection period
        for year_dcf in range(1, 6): 
            projected_fcf_nominal = latest_fcf * ((1 + sustainable_growth_rate) ** year_dcf)
            discounted_fcf_val = projected_fcf_nominal / ((1 + discount_rate) ** year_dcf)
            projected_cash_flows_dcf_list.append(discounted_fcf_val)
        
        sum_discounted_fcf = np.sum(projected_cash_flows_dcf_list)

        if discount_rate > sustainable_growth_rate:
            # Calculate FCF in the year after the projection period (Year 6 FCF for a 5-year projection)
            fcf_after_projection_nominal = latest_fcf * ((1 + sustainable_growth_rate)**(5 + 1))
            # Terminal Value = FCF_after_projection_nominal / (discount_rate - sustainable_growth_rate)
            terminal_value_nominal = fcf_after_projection_nominal / (discount_rate - sustainable_growth_rate)
            # Discount Terminal Value back to present day (discount by 5 years as TV is calculated at end of year 5)
            discounted_terminal_value = terminal_value_nominal / ((1 + discount_rate)**5)
            
            total_dcf_value = sum_discounted_fcf + discounted_terminal_value
            print(f"DCF-based valuation (FCF, {sustainable_growth_rate*100:.1f}% growth, {discount_rate*100:.1f}% discount): ${total_dcf_value:,.2f}")
            
            if market_cap > 0 and shares_outstanding and shares_outstanding > 0:
                dcf_per_share = total_dcf_value / shares_outstanding
                print(f"DCF Value per Share: ${dcf_per_share:,.2f}")
                if current_price:
                     print(f"Current Share Price: ${current_price:,.2f} (Potential upside/downside: {(dcf_per_share/current_price - 1)*100:.2f}%)")
        else:
            print("Discount rate must be greater than growth rate for terminal value calculation in DCF.")
            total_dcf_value = sum_discounted_fcf # Value without terminal value
            print(f"DCF-based valuation (sum of 5-yr discounted FCFs only): ${total_dcf_value:,.2f}")

    elif not free_cash_flow_series.empty and latest_fcf <=0:
        print(f"The company has non-positive free cash flow ({latest_fcf:,.2f}). DCF valuation based on FCF is not suitable or requires adjustments.")
    else:
        print("Free cash flow data is unavailable. Cannot perform DCF valuation.")
    
    print("--- End of Detailed Financial Analysis ---")


def plot_results(years_range, future_cash_flows_nominal, dcf_value, latest_free_cash_flow, future_stock_prices, latest_price, asset_ticker):
    """
    Plots the financial projections with enhanced visuals, including annotations.
    """
    plt.style.use('seaborn-v0_8-whitegrid') # Cleaner style
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 20)) # Slightly wider and taller
    fig.patch.set_facecolor('whitesmoke') # Light background for the figure
    
    # Main title for the entire figure
    fig.suptitle(f'{asset_ticker} - Financial Projections and Analysis', fontsize=22, fontweight='bold', y=0.99, color='#333333')

    # --- Plot 1: Stock Price and Future Price Projection ---
    color_projected_price = '#007ACC' # A nice blue
    color_current_price = '#D62728' # A contrasting red

    ax1.plot(years_range, future_stock_prices, marker='o', linestyle='-', label='Projected Stock Price', color=color_projected_price, linewidth=3, markersize=9, markeredgecolor='white', markeredgewidth=1)
    ax1.axhline(y=latest_price, color=color_current_price, linestyle='--', label=f'Current Price: ${latest_price:,.2f}', linewidth=2.5)
    ax1.set_title('Stock Price Projection vs. Current Price', fontsize=18, fontweight='bold', color='#444444')
    ax1.set_xlabel('Years from Now', fontsize=15, color='#555555')
    ax1.set_ylabel('Stock Price ($)', fontsize=15, color='#555555')
    ax1.set_xticks(years_range)
    ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
    ax1.grid(True, linestyle=':', alpha=0.6, color='gray')
    ax1.legend(fontsize=13, frameon=True, facecolor='white', edgecolor='lightgray', shadow=True)
    ax1.tick_params(axis='both', which='major', labelsize=13, colors='#555555')
    ax1.set_ylim(bottom=0) 
    # Annotate final projected price
    if future_stock_prices:
        final_proj_price = future_stock_prices[-1]
        ax1.annotate(f'Year {years_range[-1]}: ${final_proj_price:,.2f}', 
                     xy=(years_range[-1], final_proj_price), 
                     xytext=(years_range[-1] - 0.5, final_proj_price + (0.05 * final_proj_price if final_proj_price >0 else 5)), # Adjust offset
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                     fontsize=12, color='black', bbox=dict(boxstyle="round,pad=0.3", fc="aliceblue", ec="grey", lw=1))


    # --- Plot 2: DCF Analysis - Nominal Future Cash Flows and Total DCF Value ---
    color_fcf_nominal = '#2CA02C' # Green
    color_dcf_value = '#FF7F0E' # Orange

    ax2.plot(years_range, future_cash_flows_nominal, marker='s', linestyle='-', label='Projected Nominal Future FCF', color=color_fcf_nominal, linewidth=3, markersize=9, markeredgecolor='white', markeredgewidth=1)
    ax2.axhline(y=dcf_value, color=color_dcf_value, linestyle='--', label=f'Total DCF Value (Sum of Discounted FCF)', linewidth=2.5) # Shortened label
    ax2.set_title('Projected FCF (Nominal) vs. Total DCF Value', fontsize=18, fontweight='bold', color='#444444')
    ax2.set_xlabel('Year', fontsize=15, color='#555555')
    ax2.set_ylabel('Cash Flow Value ($)', fontsize=15, color='#555555')
    ax2.set_xticks(years_range)
    ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
    ax2.grid(True, linestyle=':', alpha=0.6, color='gray')
    ax2.legend(fontsize=13, frameon=True, facecolor='white', edgecolor='lightgray', shadow=True)
    ax2.tick_params(axis='both', which='major', labelsize=13, colors='#555555')
    ax2.set_ylim(bottom=min(0, (min(future_cash_flows_nominal) * 1.2) if future_cash_flows_nominal and min(future_cash_flows_nominal) < 0 else - (dcf_value * 0.1 if dcf_value > 0 else 1000) )) # Dynamic y-lim
    # Annotate DCF value
    ax2.text(years_range[0], dcf_value + (0.05 * dcf_value if dcf_value > 0 else 5000), f'DCF Value: ${dcf_value:,.2f}', 
             color=color_dcf_value, fontsize=12, fontweight='bold', ha='left', va='bottom',
             bbox=dict(boxstyle="round,pad=0.3", fc="ivory", ec="grey", lw=1))

    # --- Plot 3: Free Cash Flow Analysis - Bar chart for Nominal Future Cash Flows ---
    color_bar_fcf = '#9467BD' # Purple
    color_current_fcf = '#E377C2' # Pinkish

    ax3.bar(years_range, future_cash_flows_nominal, label='Projected Nominal FCF per Year', color=color_bar_fcf, alpha=0.85, edgecolor='black', linewidth=1)
    ax3.axhline(y=latest_free_cash_flow, color=color_current_fcf, linestyle='--', label=f'Most Recent FCF', linewidth=2.5)
    ax3.set_title('Projected Free Cash Flow (Nominal) vs. Current', fontsize=18, fontweight='bold', color='#444444')
    ax3.set_xlabel('Year', fontsize=15, color='#555555')
    ax3.set_ylabel('Free Cash Flow ($)', fontsize=15, color='#555555')
    ax3.set_xticks(years_range) 
    ax3.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
    ax3.grid(True, axis='y', linestyle=':', alpha=0.6, color='gray') 
    ax3.legend(fontsize=13, frameon=True, facecolor='white', edgecolor='lightgray', shadow=True)
    ax3.tick_params(axis='both', which='major', labelsize=13, colors='#555555')
    min_y_val_ax3 = min(0, latest_free_cash_flow * 1.1 if latest_free_cash_flow < 0 else - (max(future_cash_flows_nominal) * 0.1 if future_cash_flows_nominal and max(future_cash_flows_nominal) > 0 else 1000))
    if future_cash_flows_nominal: # Check if list is not empty
      min_y_val_ax3 = min(min_y_val_ax3, min(future_cash_flows_nominal) * 1.1 if min(future_cash_flows_nominal) < 0 else min_y_val_ax3)
    ax3.set_ylim(bottom=min_y_val_ax3)
    # Annotate Latest FCF value
    ax3.text(years_range[0], latest_free_cash_flow + (0.05 * abs(latest_free_cash_flow) if latest_free_cash_flow != 0 else 5000), f'Current FCF: ${latest_free_cash_flow:,.2f}', 
             color=color_current_fcf, fontsize=12, fontweight='bold', ha='left', va='bottom',
             bbox=dict(boxstyle="round,pad=0.3", fc="ghostwhite", ec="grey", lw=1))
    
    # Adjust layout to prevent overlap and make space for suptitle
    plt.subplots_adjust(top=0.92, hspace=0.35) # Increase hspace for more vertical room
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Alternative way to adjust, suptitle might need y adjustment
    plt.show()


def main():
    """
    Main function to run the financial analysis and plotting.
    """
    asset_ticker_input = input("Enter stock ticker (e.g., AAPL, MSFT) or press Enter for default (CVNA): ").upper()
    if not asset_ticker_input:
        asset_ticker = 'CVNA' # Default ticker
    else:
        asset_ticker = asset_ticker_input
    
    print(f"Fetching data for {asset_ticker}...")
    try:
        income_statement, cash_flow, balance_sheet, info = fetch_financial_data(asset_ticker)
    except Exception as e:
        print(f"Error fetching data for {asset_ticker}: {e}")
        print("This could be due to an invalid ticker, network issues, or changes in the yfinance API.")
        print("Please ensure the ticker is correct and you have an internet connection.")
        return

    if income_statement.empty or cash_flow.empty or balance_sheet.empty or not info:
        print(f"Could not retrieve sufficient financial data for {asset_ticker}. Analysis cannot proceed.")
        print("The ticker might be delisted, very new, or data might be unavailable for other reasons.")
        return
        
    print("Data fetched successfully. Preparing data...")
    latest_free_cash_flow, net_income_growth_rate, revenue_growth_rate, _, _, _, _, latest_net_income = prepare_data(cash_flow, income_statement, balance_sheet)

    discount_rate = 0.10 
    projection_years = 5
    
    # For DCF growth, using a sustainable fixed rate or a capped version of historical rates.
    # Using revenue_growth_rate directly can be too volatile for long-term FCF.
    # Here, we will use revenue_growth_rate but it's capped in calculate_dcf.
    # For a more robust DCF, a separate, more conservative 'sustainable_fcf_growth_rate' should be estimated.
    dcf_growth_rate_input = revenue_growth_rate 

    if latest_free_cash_flow <= 0 and dcf_growth_rate_input > 0:
        print(f"Warning: Latest Free Cash Flow is {latest_free_cash_flow:,.2f} (non-positive). Projecting with a positive growth rate ({dcf_growth_rate_input:.2%}) means FCF becomes less negative or positive.")
    elif latest_free_cash_flow <=0 and dcf_growth_rate_input <=0:
         print(f"Warning: Latest Free Cash Flow is {latest_free_cash_flow:,.2f} (non-positive). Projecting with a non-positive growth rate ({dcf_growth_rate_input:.2%}) means FCF remains or becomes more negative.")


    # Calculate DCF and nominal future cash flows
    # The dcf_value is sum of *discounted* future cash flows. future_cash_flows_nominal are *not* discounted.
    dcf_value, _, future_cash_flows_nominal = calculate_dcf(latest_free_cash_flow, discount_rate, dcf_growth_rate_input, years=projection_years)

    latest_price = info.get('currentPrice')
    if latest_price is None:
        print("Could not retrieve current price. Stock price projection will be skipped.")
        future_stock_prices = [np.nan] * projection_years # Use NaN if price not available
    else:
        # For future stock prices, using revenue growth rate as a proxy for company growth.
        # This is a simplification; stock prices are influenced by many factors.
        price_projection_growth_rate = revenue_growth_rate
        future_stock_prices = calculate_future_stock_prices(latest_price, price_projection_growth_rate, years=projection_years)
    
    years_range = np.arange(1, projection_years + 1)
    
    analyze_financials(income_statement, cash_flow, balance_sheet, info, discount_rate=discount_rate)
    
    if latest_price is not None:
        plot_results(years_range, future_cash_flows_nominal, dcf_value, latest_free_cash_flow, future_stock_prices, latest_price, asset_ticker)
    else:
        print("\nSkipping plotting due to missing current price data.")
        # Still print key calculated values if plotting is skipped
        print(f"Latest Free Cash Flow: ${latest_free_cash_flow:,.2f}")
        print(f"Calculated DCF Value (sum of discounted FCFs for {projection_years} years): ${dcf_value:,.2f}")
        if future_cash_flows_nominal:
            print(f"Projected Nominal Future Cash Flows (Years 1-{projection_years}):")
            for i, cf_nom in enumerate(future_cash_flows_nominal):
                print(f"  Year {i+1}: ${cf_nom:,.2f}")

if __name__ == "__main__":
    main()