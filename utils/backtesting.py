import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def run_backtest(analysis_data):
    """Exécute le backtest avec les paramètres actuels."""
    try:
        # Initialisation du backtest
        capital_initial = float(st.session_state.get('initial_capital', 1000000))
        cash = capital_initial
        position = 0
        portfolio_value = cash
        last_buy_price = 0.0
        peak_portfolio_value_since_buy = 0.0
        trades = []
        
        portfolio_history = pd.DataFrame(
            index=analysis_data.index,
            columns=['Cash', 'Position', 'Position Value', 'Total Value']
        )
        
        # Récupérer les paramètres de trading
        invest_percentage = st.session_state.get('invest_percentage', 100.0) / 100.0
        frais_transaction = st.session_state.get('commission_rate', 0.5) / 100.0
        stop_loss = st.session_state.get('stop_loss', 10.0) / 100.0
        take_profit = st.session_state.get('take_profit', 20.0) / 100.0
        use_trailing_stop = st.session_state.get('use_trailing_stop', True)
        trailing_stop_loss_pct = st.session_state.get('trailing_stop_pct', 5.0) / 100.0
        use_fundamental_signals = st.session_state.get('val_intrinseque', None) is not None
        marge_achat = st.session_state.get('buy_margin', 20.0) / 100.0 if use_fundamental_signals else 0.0
        marge_vente = st.session_state.get('sell_premium', 10.0) / 100.0 if use_fundamental_signals else 0.0
        val_intrinseque = st.session_state.get('val_intrinseque', None)
        tech_signal_method_active = any([
            st.session_state.get('use_mm_signal', False),
            st.session_state.get('use_rsi_signal', False),
            st.session_state.get('use_macd_signal', False)
        ])
        
        # Boucle de backtesting
        for i in range(len(analysis_data)):
            current_date = analysis_data.index[i]
            current_price = analysis_data.loc[current_date, 'Prix']
            current_open_price = analysis_data.loc[current_date, 'Ouverture']
            current_high_price = analysis_data.loc[current_date, 'Plus_Haut']
            current_low_price = analysis_data.loc[current_date, 'Plus_Bas']
            
            # Vérifier les données valides
            if pd.isna(current_price) or pd.isna(current_open_price) or pd.isna(current_high_price) or pd.isna(current_low_price):
                if i > 0:
                    previous_date = analysis_data.index[i-1]
                    portfolio_history.loc[current_date] = portfolio_history.loc[previous_date]
                else:
                    portfolio_history.loc[current_date] = [cash, position, 0.0, cash]
                continue
            
            # Initialiser la valeur du portefeuille pour aujourd'hui
            position_value = float(position) * current_price
            portfolio_value = cash + position_value
            portfolio_history.loc[current_date, ['Cash', 'Position', 'Position Value', 'Total Value']] = [cash, position, position_value, portfolio_value]
            
            # Logique de vente
            sell_signal_triggered = False
            sell_reason = ""
            execution_price = current_price
            
            if position > 0:
                # Vérifier Stop Loss
                stop_loss_price = last_buy_price * (1.0 - stop_loss)
                if current_low_price <= stop_loss_price:
                    sell_signal_triggered = True
                    sell_reason = f"Stop Loss ({stop_loss*100:.1f}%)"
                    execution_price = stop_loss_price
                
                # Vérifier Trailing Stop Loss
                if use_trailing_stop and not sell_signal_triggered:
                    peak_price_since_buy = max(last_buy_price, current_high_price if peak_portfolio_value_since_buy == 0 else peak_portfolio_value_since_buy)
                    peak_portfolio_value_since_buy = peak_price_since_buy
                    trailing_stop_price = peak_price_since_buy * (1.0 - trailing_stop_loss_pct)
                    
                    if current_low_price <= trailing_stop_price:
                        sell_signal_triggered = True
                        sell_reason = f"Trailing Stop ({trailing_stop_loss_pct*100:.1f}%)"
                        execution_price = trailing_stop_price
                
                # Vérifier Take Profit
                take_profit_price = last_buy_price * (1.0 + take_profit)
                if not sell_signal_triggered and current_high_price >= take_profit_price:
                    sell_signal_triggered = True
                    sell_reason = f"Take Profit ({take_profit*100:.1f}%)"
                    execution_price = take_profit_price
                
                # Vérifier Signal de Vente Fondamental
                if use_fundamental_signals and val_intrinseque is not None and not sell_signal_triggered:
                    sell_vi_price = val_intrinseque * (1.0 + marge_vente)
                    if current_price > sell_vi_price:
                        sell_signal_triggered = True
                        sell_reason = f"Valeur Intrinsèque (Prime {marge_vente*100:.1f}%)"
                        execution_price = current_price
                
                # Vérifier Signal de Vente Technique
                if tech_signal_method_active and not sell_signal_triggered:
                    if analysis_data.loc[current_date, 'Signal_Technique_Combine'] == -1:
                        sell_signal_triggered = True
                        sell_reason = f"Signal Technique ({st.session_state.get('tech_signal_method', '')})"
                        execution_price = current_price
            
            # Exécuter la vente
            if sell_signal_triggered:
                execution_price = max(current_low_price, min(current_high_price, execution_price))
                sell_value = float(position) * execution_price
                commission = sell_value * frais_transaction
                cash += sell_value - commission
                
                trades.append({
                    'Date': current_date, 'Type': 'Vente', 'Prix': execution_price,
                    'Quantité': position, 'Valeur': sell_value, 'Frais': commission,
                    'Cash Après': cash, 'Raison': sell_reason
                })
                
                position = 0
                last_buy_price = 0.0
                peak_portfolio_value_since_buy = 0.0
                
                # Mettre à jour la valeur du portefeuille après la vente
                position_value = 0.0
                portfolio_value = cash
                portfolio_history.loc[current_date, ['Cash', 'Position', 'Position Value', 'Total Value']] = [cash, position, position_value, portfolio_value]
            
            # Logique d'achat
            if position == 0 and not sell_signal_triggered:
                buy_signal_triggered = False
                buy_reason = ""
                execution_price = current_price
                
                # Vérifier Signal d'Achat Fondamental
                fundamental_buy = False
                if use_fundamental_signals and val_intrinseque is not None:
                    buy_vi_price = val_intrinseque * (1.0 - marge_achat)
                    if current_price < buy_vi_price:
                        fundamental_buy = True
                
                # Vérifier Signal d'Achat Technique
                technical_buy = tech_signal_method_active and analysis_data.loc[current_date, 'Signal_Technique_Combine'] == 1
                
                # Combiner les signaux d'achat
                if fundamental_buy or technical_buy:
                    buy_signal_triggered = True
                    reasons = []
                    if fundamental_buy: reasons.append(f"Valeur Intrinsèque (Marge {marge_achat*100:.1f}%)")
                    if technical_buy: reasons.append(f"Signal Technique ({st.session_state.get('tech_signal_method', '')})")
                    buy_reason = " & ".join(reasons)
                    execution_price = current_price
                
                # Exécuter l'achat
                if buy_signal_triggered:
                    amount_to_invest = cash * invest_percentage
                    
                    if execution_price > 0 and (1.0 + frais_transaction) > 0:
                        quantity_to_buy = int(amount_to_invest / (execution_price * (1.0 + frais_transaction)))
                    else:
                        quantity_to_buy = 0
                    
                    if quantity_to_buy > 0:
                        buy_value = float(quantity_to_buy) * execution_price
                        commission = buy_value * frais_transaction
                        total_cost = buy_value + commission
                        
                        if total_cost <= cash:
                            cash -= total_cost
                            position = quantity_to_buy
                            last_buy_price = execution_price
                            peak_portfolio_value_since_buy = execution_price
                            
                            trades.append({
                                'Date': current_date, 'Type': 'Achat', 'Prix': execution_price,
                                'Quantité': position, 'Valeur': buy_value, 'Frais': commission,
                                'Cash Après': cash, 'Raison': buy_reason
                            })
                            
                            # Mettre à jour la valeur du portefeuille après l'achat
                            position_value = float(position) * current_price
                            portfolio_value = cash + position_value
                            portfolio_history.loc[current_date, ['Cash', 'Position', 'Position Value', 'Total Value']] = [cash, position, position_value, portfolio_value]
        
        return portfolio_history, trades
    
    except Exception as e:
        st.error(f"Une erreur est survenue lors de l'exécution du backtest : {e}")
        st.error(traceback.format_exc())
        return pd.DataFrame(), []

def display_backtest_results(portfolio_history, trades, original_data):
    """Affiche les résultats du backtest."""
    if not portfolio_history.empty and len(portfolio_history) > 1:
        # Afficher l'évolution du portefeuille
        st.markdown("### Évolution de la Valeur du Portefeuille")
        fig_portfolio, ax_portfolio = plt.subplots(figsize=(12, 6))
        ax_portfolio.plot(portfolio_history.index, portfolio_history['Total Value'], label='Valeur Totale Portefeuille', color='green', linewidth=1.5)
        
        # Ajouter une ligne pour Buy & Hold
        common_index = original_data.index.intersection(portfolio_history.index)
        if not common_index.empty:
            first_valid_price = original_data.loc[common_index[0], 'Prix']
            if pd.notna(first_valid_price) and first_valid_price > 0:
                buy_hold_value = (st.session_state.get('initial_capital', 1000000) / first_valid_price) * original_data.loc[common_index, 'Prix']
                ax_portfolio.plot(common_index, buy_hold_value, label='Stratégie Buy & Hold', color='grey', linestyle='--', linewidth=1)
        
        ax_portfolio.set_title('Évolution de la Valeur du Portefeuille vs Buy & Hold', fontsize=14)
        ax_portfolio.set_xlabel('Date', fontsize=10)
        ax_portfolio.set_ylabel('Valeur Portefeuille (FCFA)', fontsize=10)
        ax_portfolio.grid(True, linestyle='--', alpha=0.6)
        ax_portfolio.legend(fontsize=10)
        ax_portfolio.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_portfolio)
        plt.close(fig_portfolio)
        
        # Afficher l'historique du portefeuille
        with st.expander("Historique détaillé du portefeuille (100 dernières lignes)"):
            st.dataframe(portfolio_history.tail(100).style.format({
                'Cash': '{:,.2f}',
                'Position': '{:,.0f}',
                'Position Value': '{:,.2f}',
                'Total Value': '{:,.2f}'
            }))
        
        # Afficher la liste des transactions
        st.markdown("#### Liste des Transactions")
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df['Date'] = pd.to_datetime(trades_df['Date'])
            trades_df.set_index('Date', inplace=True)
            st.dataframe(trades_df.style.format({
                'Prix': '{:.2f}',
                'Quantité': '{:.0f}',
                'Valeur': '{:.2f}',
                'Frais': '{:.2f}',
                'Cash Après': '{:.2f}'
            }))
        else:
            st.info("Aucune transaction n'a été effectuée pendant la période de backtest.")
        
        # Afficher les métriques de performance
        st.markdown("#### Métriques de Performance Clés")
        if not portfolio_history.empty:
            _display_performance_metrics(portfolio_history, trades, original_data)

def _display_performance_metrics(portfolio_history, trades, original_data):
    """Affiche les métriques de performance."""
    final_portfolio_value = portfolio_history['Total Value'].iloc[-1]
    capital_initial = float(st.session_state.get('initial_capital', 1000000))
    total_return_pct = ((final_portfolio_value / capital_initial) - 1) * 100
    
    start_date = portfolio_history.index[0]
    end_date = portfolio_history.index[-1]
    duration_days = len(portfolio_history)
    duration_years = duration_days / 252.0
    
    # Calcul du CAGR
    cagr = ((final_portfolio_value / capital_initial) ** (1 / duration_years) - 1) * 100 if duration_years > 0 and capital_initial > 0 and final_portfolio_value > 0 else 0
    
    # Volatilité Annualisée
    daily_returns = portfolio_history['Total Value'].pct_change().dropna()
    volatility_annualized = daily_returns.std() * np.sqrt(252) * 100 if not daily_returns.empty else 0.0
    
    # Ratio de Sharpe
    sharpe_ratio = 0.0
    if not daily_returns.empty and daily_returns.std() != 0:
        taux_sans_risque_annuel = st.session_state.get('risk_free_rate', 3.0) / 100.0
        risk_free_rate_daily = (1 + taux_sans_risque_annuel)**(1/252.0) - 1
        excess_returns_daily = daily_returns - risk_free_rate_daily
        if excess_returns_daily.std() != 0:
            sharpe_ratio = (excess_returns_daily.mean() / excess_returns_daily.std()) * np.sqrt(252)
    
    # Max Drawdown
    rolling_max = portfolio_history['Total Value'].cummax()
    daily_drawdown = (portfolio_history['Total Value'] / rolling_max) - 1.0
    max_drawdown = daily_drawdown.min() * 100
    
    # Performance Buy & Hold
    final_bh_value = np.nan
    bh_total_return_pct = np.nan
    bh_cagr = np.nan
    
    common_index = original_data.index.intersection(portfolio_history.index)
    if not common_index.empty:
        first_valid_price = original_data.loc[common_index[0], 'Prix']
        if pd.notna(first_valid_price) and first_valid_price > 0:
            final_bh_value = (capital_initial / first_valid_price) * original_data.loc[common_index[-1], 'Prix']
            bh_total_return_pct = ((final_bh_value / capital_initial) - 1) * 100
            bh_cagr = ((final_bh_value / capital_initial) ** (1 / duration_years) - 1) * 100 if duration_years > 0 else 0
    
    # Affichage des métriques
    col1, col2, col3 = st.columns(3)
    col1.metric("Valeur Finale Portefeuille", f"{final_portfolio_value:,.0f} FCFA", f"{total_return_pct:+.1f}% Total")
    col1.metric("Rendement Annualisé (CAGR)", f"{cagr:.1f}%")
    col2.metric("Volatilité Annualisée", f"{volatility_annualized:.1f}%")
    col2.metric("Ratio de Sharpe", f"{sharpe_ratio:.2f}")
    col3.metric("Max Drawdown", f"{max_drawdown:.1f}%")
    col3.metric("Nombre de Trades", f"{len(trades)}")
    
    st.markdown("---")
    st.markdown("### Comparaison Buy & Hold")
    col1b, col2b = st.columns(2)
    
    if pd.notna(final_bh_value):
        col1b.metric("Valeur Finale Buy & Hold", f"{final_bh_value:,.0f} FCFA", f"{bh_total_return_pct:+.1f}% Total")
        col2b.metric("CAGR Buy & Hold", f"{bh_cagr:.1f}%")
    else:
        col1b.metric("Valeur Finale Buy & Hold", "N/A")
        col2b.metric("CAGR Buy & Hold", "N/A")
